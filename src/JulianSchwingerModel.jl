module JulianSchwingerModel

import Printf, Random, Base.Iterators
import TensorOperations 
using LinearAlgebra, Base.Iterators, SHA
using LinearMaps, IterativeSolvers, TensorOperations
import Base: zero, convert # Extend base operations

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

# Metropolis for quenched field
include("metropolis.jl")

# --------------------------------------------------------------------

# Basic routine for measurements
include("measurements.jl")

# IO routines, including formatting output
include("io.jl")

########################################################################
# ----------------------------- Interface ---------------------------- #
########################################################################

# Definitions in lattice.jl
export Lattice, deepcopy!, trunc_lattice, stack_sublattice, sync!, lin2corr
# Definitions in spinor.jl
export FlatField, zero, zero!, convert, dirac_comp1, dirac_comp2
# Definitions in gamma_matrices.jl
export gamma1, gamma2, gamma5, gamma5mul!, gamma5mul, gamma5mul!
# Definitions in dirac.jl
export gamma5_Dslash_wilson_vector!, gamma5_Dslash_wilson_vector, gamma5_Dslash_linearmap
# Definitions in solvers.jl
export minres_Q
# Definitions in hmc.jl, hmc_types.jl
export HMCParam, HMCWilson_continuous_update!
# Defintions in measurements.jl
export measure_wilsonloop
# Definitions in metropolis.jl
export metropolis_update!
# Definitions in io.jl
export print_lattice, print_sep, checksum_lattice, save_lattice, load_lattice
# Definitions in randlattice.jl
export gauss, rand01, rngseed, randZ2

end # module
