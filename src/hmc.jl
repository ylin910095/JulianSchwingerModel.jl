include("./lattice.jl")
include("./spinor.jl")
include("./dirac.jl")
include("./solvers.jl")
include("./randlattice.jl")
include("./leapfrog.jl")
include("./hmc_types.jl")
include("./measurements.jl")

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
hamold is the hamiltonian of input lattice configuration. If hamold == nothing,
it will compute the old hamiltonian before integration.

Return acc::Bool for acceptance and the new hamiltonian (equal to hamold if accp == false)
"""
function HMCWilson_update!(lattice::Lattice, hmcparam::HMCParam)
    # Generate random PseudoFermion. If quenched, it will not be used
    pf = PseudoFermion(lattice, gamma5_Dslash_wilson)
    p = HMCMom(lattice)

    hamold = evalham(p, pf, lattice.quenched, lattice)

    # Leapfrog integration
    dtau = hmcparam.tau / hmcparam.nintsteps
    latticeold = deepcopy(lattice)
    leapfrog!(p, pf, hmcparam.nintsteps, dtau, lattice.quenched, lattice)

    # Accept-Reject
    hamnew = evalham(p, pf, lattice.quenched, lattice)
    deltaham = hamnew - hamold
    if rand01() < min(1, exp(-deltaham))
        accp = 1 
    else
        # Revert back lattice field
        lattice = latticeold
        accp = 0
    end
    return accp
end


function test_HMC()
    # Lattice param
    nx = 32
    nt = 32
    mass = 0.06
    beta = 5.0
    quenched = false

    # HMC param
    tau = 0.1
    integrationsteps = 20
    hmciter = 1000
    thermalizationiter = 1
    lattice = Lattice(nx, nt, mass, beta, quenched)
    hmcparam = HMCParam(tau, integrationsteps, hmciter, thermalizationiter)
    
    accptot = 0
    # Thermalization
    for ithiter in 1:hmcparam.thermalizationiter
        println("Thermalization steps: $ithiter/$(hmcparam.thermalizationiter)")
        accp = HMCWilson_update!(lattice, hmcparam)
        accptot += accp
        accprate = accptot/ithiter
        plaq = measure_wilsonloop(lattice)
        println("Accept = $accp; Acceptance rate = $accprate; Plaquette = $plaq")
    end
    # Actual measurements
    plaqsum = 0.0
    for ihmciter in 1:hmcparam.niter
        println("Measurement steps: $ihmciter/$(hmcparam.niter)")
        accp = HMCWilson_update!(lattice, hmcparam)
        accptot += accp
        accprate = accptot/(ihmciter + hmcparam.thermalizationiter)
        plaq = measure_wilsonloop(lattice)
        plaqsum += plaq
        println("Accept = $accp; Acceptance rate = $accprate; Plaquette = $plaq")
        println("Avg plaq: $(plaqsum/ihmciter)")
    end
end
test_HMC()