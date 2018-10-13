module JulianSchwinger

#############################################################################
# -------------------------- Definitions ---------------------------------- #
#############################################################################

# Set global seed for random number generators
include("randlattice.jl")

# Lattice type for gauge field and basic manipulating functions 
include("lattice.jl")

# --------------------------------------------------------------------

# spinor field type definitions and basic manipulating functions
include("spinor.jl")

# Gamma matrices definitions
include("gamma_matrices.jl")

# Dslash operators definition
include("dirac.jl")

# Solver for Dslash, now it depends on 
include("solvers.jl")

# --------------------------------------------------------------------

# Types definitions for hmc 
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

# Definitions in lattice.jl
export Lattice, deepcopy!, sync!, lin2corr
# Definitions in spinor.jl
export FlatField, zero, zero!, dirac_comp1, dirac_comp2
# Definitions found in gamma_matrices.jl
export gamma1, gamma2, gamma5, gamma5mul!, gamma5mul, gamma5mul!
# Definitions found in dirac.jl
export gamma5_Dslash_wilson_vector!, gamma5_Dslash_wilson_vector, gamma5_Dslash_linearmap
# Definitions found in hmc.jl
export HMCParam, HMCWilson_continuous_update!
# Defintions found in measurements.jl
export measure_wilsonloop
# Definitions found in io.jl
export print_lattice, checksum_lattice, save_lattice, load_lattice
# Definitions found in randlattice.jl
export gauss, rand01, rngseed

end # module
