"""
Base HMC type
"""
mutable struct HMCParam
    tau::Float64 # Total HMC evolution time
    nintsteps::Int64 # Integration timesteps
    thermalizationiter::Int64 # Number of thermalization steps
    measurements::Int64 # Number of accepted measurements
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
    field_in::FlatField, lattice::Lattice, mass::Float64
and outputs gamma5 * pseudofermion Field
"""
struct PseudoFermion
    pf::FlatField
    function PseudoFermion(lattice::Lattice, g5Dslash_vec::Any)
        pf = FlatField(undef, 2*lattice.ntot)
        Dm1pf = FlatField(undef, 2*lattice.ntot)
        for i in 1:lattice.ntot
            # Sample D^{-1}\phi according to normal distribution
            Dm1pf[dirac_comp1(i)] = (gauss() + im*gauss())/sqrt(2)
            Dm1pf[dirac_comp2(i)] = (gauss() + im*gauss())/sqrt(2)
        end
        pf = gamma5mul(g5Dslash_vec(Dm1pf, lattice, lattice.mass))
        new(pf)
    end
end
