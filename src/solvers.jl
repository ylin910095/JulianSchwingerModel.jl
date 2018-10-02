using LinearMaps
using IterativeSolvers
using Base.Iterators

include("./lattice.jl")
include("./gamma_matrices.jl")
include("./dirac.jl")
include("./spinor.jl")
include("./randlattice.jl")
include("./io.jl")
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
    # TEST1: Does minres gives desired accuracy
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
    println("TEST1: Final difference = $ddiff")


    print_sep()
    # TEST2: What happens if we change lattice, does Q changes too?
    lattice = Lattice(nx, nt, mass, beta, quenched)
    source = zero(FlatField(undef, 2*lattice.ntot))
    # Random source
    for i in 1:length(source)
        source[i] = gauss() + im * gauss()
    end
    Q = gamma5_Dslash_linearmap(lattice, lattice.mass)
    # Now change lattice
    for i in 1:lattice.ntot
        lattice.anglex[i] = 2pi * rand01()
        lattice.anglet[i] = 2pi * rand01()
    end
    sync!(lattice)
    out1 = Q * source
    out2 = gamma5_Dslash_wilson_vector(source, lattice, lattice.mass)
    ddiff = sum(out1 - out2)
    println("TEST2: Final difference = $ddiff")
    print_sep()
end


#test_minres()
