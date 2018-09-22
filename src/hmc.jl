include("./lattice.jl")
include("./spinor.jl")
include("./dirac.jl")
include("./solvers.jl")
include("./randlattice.jl")
include("./leapfrog.jl")
include("./hmc_types.jl")

"""
Calculate Hamiltonian
"""
function evalham(p::HMCMom, pf::PseudoFermion, quenched::Bool, lattice::Lattice)
    if quenched == false
        # First calculate pf contribution by inversion
        x0 = Field(undef, lattice.ntot)
        zero!(x0) # Zero initial guess
        psi = cg_Q(lattice, lattice.mass, x0, gamma5mul(pf.pf.s)) # D^{-1}phi
    end

    # Sum all parts
    ham = 0.0 
    for i in 1:lattice.ntot
        if quenched == false
            ham += SG(i,lattice) + 0.5 * p.gpx[i]*p.gpx[i] + 
                                   0.5 * p.gpt[i]*p.gpt[i] +
                                   conj(psi[i][1]) * psi[i][1] +
                                   conj(psi[i][2]) * psi[i][2]
        else
            ham += SG(i,lattice) + 0.5 * p.gpx[i]*p.gpx[i] + 
                                   0.5 * p.gpt[i]*p.gpt[i] 
        end
    end
    if real(ham) != ham
        error("Hamiltonian not real: $ham")
    end
    return real(ham)
end

"""
Perform hybrid monte-carlo update to lattice using hmcparam parameters
"""
function HMCWilson_update(lattice::Lattice, hmcparam::HMCParam)
    
    acceptno = 0
    for ihmcstep in 1:hmcparam.niters 
        println("--> HMC Steps: $ihmcstep/$(hmcparam.niters)")

        # Generate random PseudoFermion. If quenched, it will not be used
        pf = PseudoFermion(lattice, gamma5_Dslash_wilson)
        p = HMCMom(lattice)
        # Before integration, save the old hamiltonian for accept-reject step
        hamold = evalham(p, pf, hmcparam.quenched, lattice)

        # Leapfrog integration
        dtau = hmcparam.tau / hmcparam.nintsteps
        latticeold = deepcopy(lattice)
        leapfrog!(p, pf, hmcparam.nintsteps, dtau, hmcparam.quenched, lattice)

        # Accept-Reject
        hamnew = evalham(p, pf, hmcparam.quenched, lattice)
        deltaham = hamnew - hamold
        if rand01() < min(1, exp(-deltaham))
            acceptno += 1
        else
            # Revert back lattice field
            lattice = latticeold
        end
        accptrate = acceptno / ihmcstep
        println()
        println("Accept rates: $accptrate; Delta Hamiltonian : $deltaham")
        println()
    end
end


function test_HMC()
    lattice = Lattice(32, 32, -0.06, 1.0)
    hmcparam = HMCParam(0.1, 10, 100, 100, false)
    HMCWilson_update(lattice, hmcparam)
end
test_HMC()