"""
Linear index (Coulumn-major) to tuple of coordinate in (x,t)
"""
@inline function lin2corr(i::Int64, nx::Int64)::Tuple{Int64, Int64}
    return (Int64((i-1)%nx + 1),
            Int64(floor((i-1)/nx) + 1))
end

"""
Base type for 2D lattice
"""
mutable struct Lattice
    # Inputs
    nx::Int64
    nt::Int64
    mass::Float64 # Sea quarks masses
    beta::Float64
    quenched::Bool

    # Initilize derived variables
    ntot::Int64
    lin_indx::Array{Int64, 2}
    leftx::Array{Int64, 2}
    rightx::Array{Int64, 2}
    upt::Array{Int64, 2}
    downt::Array{Int64, 2}
    corr_indx::Array{Tuple{Int64, Int64}, 2} # coordinate tuple in the order of (x,t)

    # Gauge link and gauge angles
    anglex::Array{Float64}
    anglet::Array{Float64}
    linkx::Array{ComplexF64, 2}
    linkt::Array{ComplexF64, 2}

    # Inner constructor method to initilize structure
    # If anglex0/anglet0 != nothing, the initial lattice will be constructed
    # from the given input
    function Lattice(nx::Int64, nt::Int64, mass::Float64, beta::Float64, quenched::Bool,
                     anglex0=nothing, anglet0=nothing)
        ntot = nx * nt
        leftx  = Array{Int64, 2}(undef, nx, nt)
        rightx = Array{Int64, 2}(undef, nx, nt)
        upt = Array{Int64, 2}(undef, nx, nt)
        downt = Array{Int64, 2}(undef, nx, nt)
        corr_indx = Array{Tuple{Int64, Int64}, 2}(undef, nx, nt)
        lin_indx = Array{Int64, 2}(undef, nx, nt)

        # Gauge stuff
        anglex = Array{Float64, 2}(undef, nx, nt)
        anglet = Array{Float64, 2}(undef, nx, nt)
        linkx = Array{ComplexF64, 2}(undef, nx, nt)
        linkt = Array{ComplexF64, 2}(undef, nx, nt)

        for i in 1:ntot
            # Find the closest neighbors and coordinate indices
            lin_indx[i] = i
            if (i-1) % nx != 0
                leftx[i] = i - 1
            else
                leftx[i] = i + nx - 1
            end
            if i % nx != 0
                rightx[i] = i + 1
            else
                rightx[i] = i - nx + 1
            end
            upt[i] = i + nx
            if upt[i] > ntot
                upt[i] -= ntot
            end
            downt[i] = i - nx
            if downt[i] < 1
                downt[i] += ntot
            end
            corr_indx[i] = lin2corr(i, nx)

            # Initilize gauge links
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
        # The ordering here corresponds to the ordering of definition
        # at the beginning of struct. If wrongly ordered, it will raise
        # errors
        new(nx, nt, mass, beta, quenched,
            ntot, lin_indx, leftx, rightx, upt, downt, corr_indx,
            anglex, anglet, linkx, linkt)
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
    for i in 1:lattice_to.ntot
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
Called after updating gauge angles to make gauge links consistent
"""
function sync!(lattice::Lattice)
    for i in lattice.ntot
        lattice.linkx[i] = exp(lattice.anglex[i]*im)
        lattice.linkt[i] = exp(lattice.anglet[i]*im)
    end
end