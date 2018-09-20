using LinearMaps
using IterativeSolvers

include("./lattice.jl")
include("./gamma_matrices.jl")
include("./dirac.jl")
include("./spinor.jl")
"""
Q = gamma5_Dslash_wilson
Solve Qx = source using conjugate gradient
x will be updated in-place at the end
"""
function cg_Q(lattice::Lattice, mass::Float64, x0::Spinor, source::Spinor)
    y = unravel(source.s)
    x0 = unravel(x0.s)
    Q = gamma5_Dslash_linearmap(lattice, mass)
    cg!(x0, Q, y; tol=1e-30, verbose=true)
    return Spinor(lattice.ntot, ravel(x0))
end

function test_cg()
    nx = 3
    nt = 3
    mass = 0.1
    beta = 2.0
    lattice = Lattice(nx, nt, mass, beta)
    source = Spinor(lattice.ntot)
    x0 = Spinor(lattice.ntot)
    # Random source
    for i in 1:lattice.ntot
        source.s[i] = [randn(Float64) + im*randn(Float64), 
                       randn(Float64) + im*randn(Float64)]
    end
    spinor_out = cg_Q(lattice, mass, x0, source)

    # Test to see if the solutions have converged
    y = gamma5_Dslash_wilson(spinor_out, lattice, mass)
    ddiff = 0
    for i in 1:lattice.ntot
        ddiff += y.s[i][1] - source.s[i][1]
        ddiff += y.s[i][2] - source.s[i][2]
    end
    """
    display(unravel(y.s))
    println()
    display(unravel(source.s))
    println()
    """
    println("Final difference: $ddiff")
end

test_cg()