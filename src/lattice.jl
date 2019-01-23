"""
Linear index (Coulumn-major) to tuple of coordinate in (x,t)
"""
@inline function lin2corr(i::Int64, nx::Int64)::Tuple{Int64, Int64}
    return (Int64((i-1)%nx + 1),
            Int64(floor((i-1)/nx) + 1))
end

"""
Base type for 2D lattice

Types of boundary conditions supported:
    "periodic": self-explanatory

        OR

    2D array of Float64 with length (2*nx) in each t dimension: 
        Dirichlet boundary condition with one padding (including 4 corners)
        in time direction ONLY. Spatial direction is still periodic
        Values of array are interpreted as following (number lables the index
        of this array, going counterclockwise). "o" 's denote all the ghost cells 
        and "." 's are the actually lattice points.

        For example, here we have a (nx=4, nt=3) lattice.
        "o" 's denote all the ghost cells  and "." 's are the actually 
        lattice points. Number lables the index of this array, going 
        counterclockwise

        increasing t-direction ---->

         1  o  .  .  .  .  o  5       
         2  o  .  .  .  .  o  6     (point down arrow) 
         3  o  .  .  .  .  o  7    increasing x-direction
         4  o  .  .  .  .  o  8  


    TODO: Reimplement Lattice with custom indexing (using getindex)
          so the layout in the memory is maximally efficient for the
          accessing pattern of Dslash operator.
"""
mutable struct Lattice
    # Inputs
    nx::Int64
    nt::Int64
    mass::Float64 # sea quarks masses
    beta::Float64
    boundary_cond::String # boundary condition
    quenched::Bool

    # Initilize derived variables
    ntot::Int # total number of lattice sites EXCLUDING ghost cells
    ntot_ghost::Int # total number of lattice sites INCLUDING ghost cells
    lin_indx::Array{Int, 2}
    leftx::Array{Int, 2}
    rightx::Array{Int, 2}
    upt::Array{Int, 2}
    downt::Array{Int, 2}
    corr_indx::Array{Tuple{Int, Int}, 2} # coordinate tuple in the order of (x,t)

    # Gauge link and gauge angles
    anglex::Array{Float64}
    anglet::Array{Float64}
    linkx::Array{ComplexF64, 2}
    linkt::Array{ComplexF64, 2}

    # Boundary phases for fermions used by Dslash operation
    # fermibc[1] is the phase for downt direction while
    # fermibc[2] is the phase for upt direction
    fermibc::Array{ComplexF64, 2}

    # Inner constructor method to initilize structure
    # If anglex0/anglet0 != nothing, the initial lattice will be constructed
    # from the given input
    function Lattice(nx::Int64, nt::Int64, mass::Float64, beta::Float64, quenched::Bool; 
                     anglex0=nothing, anglet0=nothing, boundary_cond="periodic",
                     fermibc=nothing)
        ntot = nx * nt
        ntot_ghost = ntot + (2*nx)
        # Append one extra length to time dimension to accommodate to 
        # different boundary conditions (ghost cells)
        leftx  = Array{Int64, 2}(undef, nx, nt+2)
        rightx = Array{Int64, 2}(undef, nx, nt+2)
        upt = Array{Int64, 2}(undef, nx, nt+2)
        downt = Array{Int64, 2}(undef, nx, nt+2)
        corr_indx = Array{Tuple{Int64, Int64}, 2}(undef, nx, nt+2)
        lin_indx = Array{Int64, 2}(undef, nx, nt+2)
        anglex = Array{Float64, 2}(undef, nx, nt+2)
        anglet = Array{Float64, 2}(undef, nx, nt+2)
        linkx = Array{ComplexF64, 2}(undef, nx, nt+2)
        linkt = Array{ComplexF64, 2}(undef, nx, nt+2)

        # Check which bc we want: list of list or string
        @assert (typeof(boundary_cond) == Array{Float64,2} 
                 || boundary_cond == "periodic") "Unknown boundary condition" * 
                 " (did you use Int instead of Float64?)"
        @assert (fermibc == nothing
                || size(fermibc)[1] == ntot) "fermibc has wrong size!"
        # TODO: Check sanity of fermibc (fermibc has to be nothing if periodic bc)

        # Loop over all non-ghost cells
        for i in 1:ntot
            # Find the closest neighbors
            lin_indx[i] = i
            leftx[i] = i - 1 
            rightx[i] = i + 1 
            upt[i] = i + nx
            downt[i] = i - nx
            
            # Take care of bc
            if upt[i] > ntot 
                if boundary_cond == "periodic"
                    upt[i] -= ntot # periodic
                else # dirichlet
                    upt[i] = ntot + nx + (i-1)%nx + 1
                end
            end
            if downt[i] < 1
                if boundary_cond == "periodic"
                    downt[i] += ntot # periodic
                else #dirichlet
                    downt[i] = ntot + (i-1)%nx + 1 
                end
            end

            # Always periodic in space at the moment
            if (i-1) % nx == 0
                leftx[i] = i + nx - 1 # periodic
            end
            if i % nx == 0
                rightx[i] = i - nx + 1 # periodic
            end

            # Find the corresponding coordinate indices
            corr_indx[i] = lin2corr(i, nx)

            # Initilize gauge links to unity
            # Only do it for non-ghost cells
            if anglex0 != nothing
                anglex[i] = anglex0[i]
            else
                anglex[i] = 0.0
            end
            if anglet0 != nothing
                anglet[i] = anglet0[i]
            else
                anglet[i] = 0.0
            end

            # Calculate links
            linkx[i] = exp(anglex[i]*im)
            linkt[i] = exp(anglet[i]*im)
        end
        # We do not access ghost cells in periodic bc so
        # set to nothing to avoid mistaken access

        # note: Julia don't accept NaN or missing for 
        # integer arrays. Use -99 as placeholder for errorons
        # index as Julia doesn't accept negative indexing (v1)
        errint = -99
        if boundary_cond == "periodic"
            for i in ntot+1:ntot_ghost
                # The coodinate indices for ghosts cells are always nothing
                # to avoid accidental mix-up
                corr_indx[i] = (errint, errint)
                lin_indx[i] = errint
                leftx[i] = errint
                rightx[i] = errint 
                upt[i] = errint
                downt[i] = errint
                anglex[i] = NaN
                anglet[i] = NaN
                linkx[i] = NaN
                linkt[i] = NaN
            end
        else # dirichlet
        # Hacky implmentation by going through each 
        # padding direction separately

        # First ignore four corners and going from bottom to top
        # around the lattice as depicted in the picture 
            for i in ntot+1:ntot_ghost
                # Should not access ghost cells' coordinate indices
                corr_indx[i] = (errint, errint)
                lin_indx[i] = i 
                anglex[i] = boundary_cond[1,i-ntot]
                anglet[i] = boundary_cond[2,i-ntot]
                linkx[i] = exp(anglex[i]*im)
                linkt[i] = exp(anglet[i]*im)

                # Setting up neighors in x-direction
                # Always periodic in space direction
                leftx[i] = i - 1
                rightx[i] = i + 1
                if (i-1) % nx == 0
                    leftx[i] = i + nx - 1
                end
                if i % nx == 0
                    rightx[i] = i - nx + 1
                end
            end

            # Now find the neighbors in t-direction
            # ghost cells below t=0 timeslice
            for i in ntot+1:ntot+nx
                downt[i] = errint # boundary
                upt[i] = (i-1)%nx +1
            end
            # Ghost cells above t=nt timeslice
            for i in ntot+nx+1:ntot_ghost
                upt[i] = errint # boundary
                downt[i] = ((i-1)%nx + 1) + (nt-1)*nx
            end
        end # boundary condition

        # Rename variable to string if not periodic
        if boundary_cond != "periodic"
            boundary_cond = "time dirichlet"
        end

        # Finally, default to fermion anti-periodic boundary phases if not given 
        if fermibc == nothing
            fermibc = Array{ComplexF64, 2}(undef, ntot, 2)
            for i in 1:ntot
                fermibc[i, 1] = 1
                fermibc[i, 2] = 1
                if corr_indx[i][2] == 1
                    fermibc[i, 1] = -1
                end
                if corr_indx[i][2] == nt
                    fermibc[i, 2] = -1
                end
            end
        end

        # The ordering here corresponds to the ordering of definition
        # at the beginning of struct. If wrongly ordered, it will raise
        # errors
        new(nx, nt, mass, beta, boundary_cond, quenched, 
            ntot, ntot_ghost, lin_indx, leftx, rightx, upt, 
            downt, corr_indx, anglex, anglet, linkx, linkt, fermibc)
    end
end

"""
Copy all lattice content from lattice_from to lattice_to
"""
function deepcopy!(lattice_to::Lattice, lattice_from::Lattice)
    lattice_to.ntot = lattice_from.ntot
    lattice_to.nx = lattice_from.nx
    lattice_to.nt = lattice_from.nt
    lattice_to.mass = lattice_from.mass
    lattice_to.beta = lattice_from.beta
    lattice_to.quenched = lattice_from.quenched
    lattice_to.boundary_cond = lattice_from.boundary_cond
    lattice_to.fermibc = lattice_from.fermibc

    # Copy everything including ghost cells
    for i in 1:length(lattice_from.anglex)
        lattice_to.lin_indx = lattice_from.lin_indx
        lattice_to.leftx = lattice_from.leftx
        lattice_to.rightx = lattice_from.rightx
        lattice_to.upt = lattice_from.upt
        lattice_to.downt = lattice_from.downt
        lattice_to.corr_indx = lattice_from.corr_indx
        lattice_to.anglex = lattice_from.anglex
        lattice_to.anglet = lattice_from.anglet
        lattice_to.linkx = lattice_from.linkx
        lattice_to.linkt = lattice_from.linkt
    end
end

"""
Given a lattice A, return a sublattice B in A specified by the timeslices
t_range with time dirichlet boundary condition. The range is inclusive at
both ends
TODO: Option to preserve original coor_indx instead of recreating it?
"""
function trunc_lattice(lattice::Lattice, t_range::UnitRange{Int})
    nt = length(t_range)
    nx = lattice.nx
    tmin = minimum(t_range)
    tmax = maximum(t_range)
    @assert tmin >= 1 "t_range must have minimum"*
                      "index greater than 1"
    @assert nt < lattice.nt "length(t_range) must be smaller than the orignal"*
                            "lattice size in t direction!"

    # Linear indiced of t_range and initial lattice angles
    imin = (tmin-1)*nx + 1
    imax = imin + nx*(tmax-tmin+1) - 1
    anglex0 = lattice.anglex[imin:imax]
    anglet0 = lattice.anglet[imin:imax]

    # Setting up boundary condition ghost cells
    bc = Array{Float64, 2}(undef, 2, lattice.nx*2)

    for i in imin:imin+nx-1
        gind = i - imin + 1 # ghost cell index
        bc[1, gind] = lattice.anglex[lattice.downt[i]]
        bc[2, gind] = lattice.anglet[lattice.downt[i]]
    end

    for i in imax-nx+1:imax
        gind = i - imax + 2*nx # ghost cell index
        bc[1, gind] = lattice.anglex[lattice.upt[i]]
        bc[2, gind] = lattice.anglet[lattice.upt[i]]
    end
    
    # Retrieve fermibc
    fermibc = lattice.fermibc[imin:imax,:]

    # Create new lattice
    sublattice = Lattice(lattice.nx, nt, lattice.mass, lattice.beta, 
                         lattice.quenched, 
                         anglex0=anglex0, anglet0=anglet0,
                         boundary_cond=bc, fermibc=fermibc)
    return sublattice
end

"""
Stack two sublattices together
If periodic == true, then the ghost cells will not be used 
and the resulting lattice will be periodic (useful for reconstructing the
whole lattice from domains)
TODO: Option to preserve original coor_indx instead of recreating it?
"""
function stack_sublattice(downlattice::Lattice, 
                          uplattice::Lattice, 
                          periodic=false,
                          safety_check=true)::Lattice

    @assert downlattice.boundary_cond == "time dirichlet" "Can only stack sublattices "*
                            "with time dirichlet bc!"
    @assert uplattice.boundary_cond == "time dirichlet" "Can only stack sublattices "*
                            "with time dirichlet bc!"

    # TODO: Do other safety checks such as consistent nx dimension, mass
    #       using a function (also include check function to trunc_lattice function)

    # Check if the ghost cells match
    # Bottom check
    if (periodic == false && safety_check == true)
        for i in uplattice.ntot+1:uplattice.ntot+uplattice.nx
            @assert uplattice.anglex[i] == downlattice.anglex[i-uplattice.ntot+
                                                            downlattice.ntot-downlattice.nx]
                "Boundaries do not match!"
            @assert uplattice.anglet[i] == downlattice.anglet[i-uplattice.ntot+
                                                            downlattice.ntot-downlattice.nx]
                "Boundaries do not match!"   
        end
        # Top check
        for i in downlattice.ntot+downlattice.nx+1:downlattice.ntot_ghost
            @assert (downlattice.anglex[i] == 
            uplattice.anglex[i-downlattice.nx-downlattice.ntot]) "Boundaries do not match!"
            @assert (downlattice.anglet[i] == 
            uplattice.anglet[i-downlattice.nx-downlattice.ntot]) "Boundaries do not match!"   
        end
    end
    
    # Start stacking
    nx_stack = uplattice.nx
    nt_stack = uplattice.nt + downlattice.nt
    ntot_stack = nx_stack * nt_stack
    mass_stack = uplattice.mass
    quenched_stack = uplattice.quenched
    beta_stack = uplattice.beta

    # Setting up angles
    anglex0 = Array{Float64, 2}(undef, nx_stack, nt_stack+2)
    anglet0 = Array{Float64, 2}(undef, nx_stack, nt_stack+2)
    anglex0[1:downlattice.ntot] = downlattice.anglex[1:downlattice.ntot]
    anglet0[1:downlattice.ntot] = downlattice.anglet[1:downlattice.ntot]
    anglex0[downlattice.ntot+1:ntot_stack] = uplattice.anglex[1:uplattice.ntot]
    anglet0[downlattice.ntot+1:ntot_stack] = uplattice.anglet[1:uplattice.ntot]

    if periodic == false
        # Setting up boundary condition ghost cells
        bc = Array{Float64, 2}(undef, 2, nx_stack*2)
        # Bottom
        dmin = downlattice.ntot+1 
        dmax = downlattice.ntot+downlattice.nx
        bc[1, 1:nx_stack] = downlattice.anglex[dmin:dmax]
        bc[2, 1:nx_stack] = downlattice.anglet[dmin:dmax]
        # Top
        umin = uplattice.ntot_ghost - uplattice.nx + 1
        umax = uplattice.ntot_ghost
        bc[1, nx_stack+1:2*nx_stack] = uplattice.anglex[umin:umax]
        bc[2, nx_stack+1:2*nx_stack] = uplattice.anglet[umin:umax]

        # Create appropriate fermi boundary phase when stacking
        # vcat concatenate along the first dimension
        fermibc = vcat(downlattice.fermibc, uplattice.fermibc)
    else
        bc = "periodic"
        fermibc = nothing
    end

    # Create stacked lattice
    stack_lattice = Lattice(nx_stack, nt_stack, mass_stack, beta_stack, 
                            quenched_stack, 
                            anglex0=anglex0, anglet0=anglet0,
                            boundary_cond=bc, fermibc=fermibc)
    return stack_lattice
end

"""
Called after updating gauge angles to make gauge links consistent
"""
function sync!(lattice::Lattice)
    # Sync all links including links in ghost cells
    for i in 1:lattice.ntot_ghost
        lattice.linkx[i] = exp(lattice.anglex[i]*im)
        lattice.linkt[i] = exp(lattice.anglet[i]*im)
    end
end