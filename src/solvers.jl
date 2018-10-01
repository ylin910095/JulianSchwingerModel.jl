using LinearMaps
using IterativeSolvers
using Base.Iterators

include("./lattice.jl")
include("./gamma_matrices.jl")
include("./dirac.jl")
include("./spinor.jl")
include("./randlattice.jl")
"""
Q = gamma5_Dslash_wilson
Solve Qx = source using conjugate gradient
return flatten Field
"""
function minres_Q(Q::Any, lattice::Lattice, mass::Float64,
              source::FlatField, verbose::Bool=false)
    sol = minres(Q, source; tol=1e-16, verbose=verbose)
    return sol
end

function test_minres()
    nx = 10
    nt = 10
    mass = 0.1
    beta = 2.0
    quenched = false
    lattice = Lattice(nx, nt, mass, beta, quenched)
    source = zero(FlatField(undef, 2*lattice.ntot))
    # Random source
    for i in 1:length(source)
        source[i] = gauss() + im * gauss()
    end
    # Constructing linear map
    Q = gamma5_Dslash_linearmap(lattice, lattice.mass)
    field_out = minres_Q(Q, lattice, mass, source, false)

    # Test to see if the solutions have converged
    y = gamma5_Dslash_wilson_vector(field_out, lattice, mass)
    ddiff = sum(y - source)
    println("Final difference: $ddiff")
end

#test_minres()
