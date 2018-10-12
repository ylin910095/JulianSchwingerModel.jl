module JulianSwinger

#############################################################################
# -------------------------- Definitions ---------------------------------- #
#############################################################################
# Set global seed for random number generators
include("randlattice.jl")

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
# Include common operators on lattice structure defined below
include("lattice.jl")

# --------------------------------------------------------------------

"""
Base type for spinor fields. FlatField is a vector of 
length 2*lattice.ntot where index 2*(i-1) + 1 is the first Dirac
component at lattice site i, and 2*(i-1) + 2 is the second Dirac
component at lattice site i.
"""
FlatField = Vector{ComplexF64}

# Basic operators on spinor type defined below
include("spinor.jl")

# Gamma matrices definitions
include("gamma_matrices.jl")

# Dslash operators definition
include("dirac.jl")

# Solver for Dslash, now it depends on 
include("solvers.jl")

# --------------------------------------------------------------------

"""
Base HMC type
"""
mutable struct HMCParam
    tau::Float64 # Total HMC evolution time
    nintsteps::Int64 # Integration timesteps
    thermalizationiter::Int64 # Number of thermalization steps
    measurements::Int64 # Number of accepted measurements
end

# Types definition of hmc momenta and pseudofermion field
include("hmc_types.jl")

# Unimproved gauge action and hmc wilson forces
include("hmc_wilson_forces.jl")

# Integrator
include("leapfrog.jl")

# HMC implementation
include("hmc.jl")

# --------------------------------------------------------------------

# Basic routine for measurements
include("measurements.jl")

# IO routines, including formatting output
include("io.jl")



########################################################################
# ----------------------------- Interface ---------------------------- #
########################################################################

# Definitions found above and in lattice.jl
export Lattice, deepcopy!, sync!, lin2corr
# Definitions found above and in spinor.jl
export FlatField, zero, zero!, dirac_comp1, dirac_comp2
# Definitions found in gamma_matrices.jl
export gamma1, gamma2, gamma5, gamma5mul!, gamma5mul, gamma5mul!
# Definitions found in dirac.jl
export gamma5_Dslash_wilson_vector!, gamma5_Dslash_wilson_vector, gamma5_Dslash_linearmap
# Definitions found above and in hmc.jl
export HMCParam, HMCWilson_continuous_update!
# Defintions found in measurements.jl
export measure_wilsonloop
# Definitions found in io.jl
export print_lattice, checksum_lattice, save_lattice, load_lattice
# Definitions found in randlattice.jl
export gauss, rand01, rngseed

end # module