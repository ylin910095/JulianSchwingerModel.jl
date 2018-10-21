
include("../src/lattice.jl")
include("../src/dirac.jl")
include("../src/spinor.jl")
include("../src/solvers.jl")
include("../src/io.jl")
include("../src/randlattice.jl")
include("../src/gamma_matrices.jl")

"""
Generate a set of lattice and field to be used by
all test routines
"""
function make_bmfields()
    # Gauge parameters
    nx = 32
    nt = 32
    # Hopping parameter = (2m0 + 4r)^{-1} at 2D where r is Wilson parameter
    kappa = 0.22
    mass = (kappa^-1 - 4)/2
    beta = 6.0
    quenched = false
    # Initilize some random lattice and sources for the solver
    lattice = Lattice(nx, nt, mass, beta, quenched)
    source = FlatField(undef, 2 * lattice.ntot)
    ffield_out = zero(FlatField(undef, 2 * lattice.ntot))
    for i in 1:lattice.ntot
        lattice.anglex[i] = rand01() * 2 * pi
        lattice.anglet[i] = rand01() * 2 * pi
        lattice.linkx[i] = exp(lattice.anglex[i])
        lattice.linkt[i] = exp(lattice.anglet[i])
        source[dirac_comp1(i)] = gauss() + im * gauss() 
        source[dirac_comp2(i)] = gauss() + im * gauss()
    end
    return lattice, source, ffield_out
end

function benchmark_minres()
    bmiter = 50 # Number of test iterations
    lattice, source, ffield_out = make_bmfields()
    Q = gamma5_Dslash_linearmap(lattice, lattice.mass)
    print_lattice(lattice)
    println("Benchmark: minres_Q ($bmiter iterations)")
    @time for i in 1:bmiter
        ans = minres_Q(Q, lattice, lattice.mass, gamma5mul(source))
    end
    print_sep()
end

# Benchmark the speed of Wilson operator of various implementations
function benchmar_dirac()
    bmiter = 10000 # Number of test iterations
    lattice, source, ffield_out = make_bmfields()
    print_lattice(lattice)
    println("Benchmark: gamma5_Dslash_wilson ($bmiter iterations)")
    @time for i in 1:bmiter
        gamma5_Dslash_wilson(source, lattice, lattice.mass)
    end
    print_sep()
    println("Benchmark: gamma5_Dslash_linearmap ($bmiter iterations)")
    A = gamma5_Dslash_linearmap(lattice, lattice.mass)
    @time for i in 1:bmiter
        c = A * source
    end
    print_sep()
    println("Benchmark: gamma5_Dslash_wilson! ($bmiter iterations)")
    @time for i in 1:bmiter
        gamma5_Dslash_wilson!(ffield_out, source, lattice, lattice.mass)
    end

end

#benchmark_minres()
benchmar_dirac()
