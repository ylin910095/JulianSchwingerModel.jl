include("./lattice.jl")
include("./spinor.jl")
include("./gamma_matrices.jl")
include("./randlattice.jl")

mutable struct HMCParam
    tau::Float64 # Total HMC evolution time
    nintsteps::Int64 # Integration timesteps
    niter::Int64 # Number of steps for HMC
    thermalizationiter::Int64 # Number of thermalization steps
end

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
struct PseudoFermion
    pf::Spinor
    function PseudoFermion(lattice::Lattice, g5Dslash::Any)
        pf = Spinor(lattice.ntot)
        Dm1pf = Spinor(lattice.ntot)  
        for i in 1:lattice.ntot
            # Sample D^{-1}\phi according to normal distribution
            Dm1pf.s[i] = [(gauss() + im*gauss())/sqrt(2), (gauss() + im*gauss())/sqrt(2)]
        end
        pf.s = gamma5mul(g5Dslash(Dm1pf.s, lattice, lattice.mass))
        new(pf)
    end
end
