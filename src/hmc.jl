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
function evalham(pf::PseudoFermion, p::HMCMom, lattice::Lattice)
    ham = 0.0 
    for i in 1:lattice.ntot
        ham += SG(i,lattice) + p.gpx[i]*p.gpx[i] + p.gpt[i]*p.gpt[i]
    end
    if real(ham) != ham
        error("Not real hamiltonian: $ham")
    end
    return real(ham)
end

"""
Perform hybrid monte-carlo update to lattice using hmcparam parameters
"""
function HMCWilson_update(lattice::Lattice, hmcparam::HMCParam)
    # Generate random gauge momenta 
    p = HMCMom(lattice)
    # Generate random PseudoFermion if not quenched
    if hmcparam.quenched == false
        acceptno = 0
        for ihmcstep in 1:hmcparam.niters 
            println("--> HMC Steps: $ihmcstep/$(hmcparam.niters)")
            pf = PseudoFermion(lattice, gamma5_Dslash_wilson)

            # Before integration, save the old hamiltonian for accept-reject step
            hamold = evalham(pf, p, lattice)
            
            # Leapfrog integration
            dtau = hmcparam.tau / hmcparam.nintsteps
            latticeold = deepcopy(lattice)
            leapfrog!(p, pf, hmcparam.nintsteps, dtau, lattice)

            # Accept-Reject
            hamnew = evalham(pf, p, lattice)
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
    else
        # Imeplement Metropolis here or somewhere
        error("Quenched update not implemented yet")
    end
end


function test_HMC()
    lattice = Lattice(32, 32, -0.0600, 1.0)
    hmcparam = HMCParam(1, 10, 10, 100, false)
    HMCWilson_update(lattice, hmcparam)
end
test_HMC()