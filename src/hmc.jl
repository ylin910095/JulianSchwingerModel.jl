include("./lattice.jl")
include("./spinor.jl")
include("./dirac.jl")
include("./solvers.jl")
include("./randlattice.jl")

mutable struct HMCMom
    gpx::Array{Float64}
    gpt::Array{Float64}

    # Intialize Gaussian random field
    function HMCMom(lattice::Lattice)
        gpx = Array{Float64}(undef, lattice.nx, lattice.nt)
        gpt = Array{Float64}(undef, lattice.nx, lattice.nt)
        for i in 1:lattice.ntot
            gpx[i], gpt[i] = [gauss(), gauss()] 
        end
        new(gpx, gpt)
    end
end

mutable struct HMCParam
    tau::Float64 # Total HMC evolution time
    nsteps::Int64 # Integration timesteps
    quenched::Bool # Quenched or not
end

"""
Gamma5PseudoFermion field will be generated according to PDF:

P(phi) = N*exp(D^{-1}phi, D^{-1}phi) 

where phi is PF field, D is the Dslash_wilson operator,
and N is some normalization factor.

g5Dslash input for the inner constructor is the 
gamma5 * Dslash operator
in which has expect arguments:
    field_in::Field, lattice::Lattice, mass::Float64
and outputs gamma5 * pseudofermion Field
"""
mutable struct Gamma5PseudoFermion
    g5pf::Spinor
    Dm1pf::Spinor
    function Gamma5PseudoFermion(lattice::Lattice, g5Dslash::Any)
        g5pf = Spinor(lattice.ntot)
        Dm1pf = Spinor(lattice.ntot)
        for i in lattice.ntot
            # Sample D^{-1}\phi according to normal distribution
            Dm1pf.s[i] = [gauss() + im*gauss(), gauss() + im*gauss()]
        end
        g5pf.s = g5Dslash(Dm1pf.s, lattice, lattice.mass)
        new(g5pf, Dm1pf)
    end
end

"""
Perform hybrid monte-carlo update to lattice using hmcparam parameters
"""
function HMCWilson_update(lattice::Lattice, hmcparam::HMCParam)
    # Generate random gauge momenta 
    p = HMCMom(lattice)
    # Generate random Gamma5PseudoFermion if not quenched
    if hmcparam.quenched == false
        g5pf = Gamma5PseudoFermion(lattice, gamma5_Dslash_wilson)
    end
end



function test_HMC()
    lattice = Lattice(3, 3, 0.1, 0.1)
    hmcparam = HMCParam(1, 100, false)
    HMCWilson_update(lattice, hmcparam)
end
test_HMC()