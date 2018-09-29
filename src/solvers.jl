using LinearMaps
using IterativeSolvers

include("./lattice.jl")
include("./gamma_matrices.jl")
include("./dirac.jl")
include("./spinor.jl")
include("./randlattice.jl")
"""
Q = gamma5_Dslash_wilson
Solve Qx = source using conjugate gradient
return spinor answer
If a linear operator Q is given, it will use it to solve the system
"""
function cg_Q(lattice::Lattice, mass::Float64, source::Field,
              verbose::Bool=false)
    y = unravel(source)
    Q = gamma5_Dslash_linearmap(lattice, mass)
    sol = cg(Q, y; tol=1e-16, verbose=verbose)
    return ravel(sol)
end

function test_cg()
    nx = 3
    nt = 3
    mass = 0.1
    beta = 2.0
    quenched = false
    lattice = Lattice(nx, nt, mass, beta, quenched)
    source = Spinor(lattice.ntot)
    x0 = Spinor(lattice.ntot)
    # Random source
    for i in 1:lattice.ntot
        source.s[i] = [gauss() + im*gauss(),
                       gauss() + im*gauss()]
    end
    field_out = cg_Q(lattice, mass, x0.s, source.s)

    # Test to see if the solutions have converged
    y = gamma5_Dslash_wilson(field_out, lattice, mass)
    ddiff = 0
    for i in 1:lattice.ntot
        ddiff += y[i][1] - source.s[i][1]
        ddiff += y[i][2] - source.s[i][2]
    end
    """
    display(unravel(y.s))
    println()
    display(unravel(source.s))
    println()
    """
    println("Final difference: $ddiff")
end

#test_cg()
