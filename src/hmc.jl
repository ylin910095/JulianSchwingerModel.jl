using Printf

include("./lattice.jl")
include("./spinor.jl")
include("./dirac.jl")
include("./solvers.jl")
include("./randlattice.jl")
include("./leapfrog.jl")
include("./hmc_types.jl")
include("./measurements.jl")
include("./io.jl")

"""
Calculate Hamiltonian. Q is returned by gamma5_Dslash_linearmap
"""
function evalham(Q::Any, p::HMCMom, pf::PseudoFermion, lattice::Lattice)
    if lattice.quenched == false
        # First calculate pf contribution by inversion
        psi = minres_Q(Q, lattice, lattice.mass, gamma5mul(pf.pf)) # D^{-1}phi
    end

    # Sum all parts
    ham = 0.0
    for i in 1:lattice.ntot
        if lattice.quenched == false
            ham += SG(i,lattice) + 0.5 * p.gpx[i]*p.gpx[i] +
                                   0.5 * p.gpt[i]*p.gpt[i] +
                                   conj(psi[dirac_comp1(i)]) * psi[dirac_comp1(i)] +
                                   conj(psi[dirac_comp2(i)]) * psi[dirac_comp2(i)]
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
    pf = PseudoFermion(lattice, gamma5_Dslash_wilson_vector)
    p = HMCMom(lattice)

    # Generate Q linear operator that will be used throughout
    Q = gamma5_Dslash_linearmap(lattice, lattice.mass)

    hamold = evalham(Q, p, pf, lattice)

    # Leapfrog integration
    dtau = hmcparam.tau / hmcparam.nintsteps
    latticeold = deepcopy(lattice)
    leapfrog!(Q, p, pf, hmcparam.nintsteps, dtau, lattice)

    # Accept-Reject
    hamnew = evalham(Q, p, pf, lattice)
    deltaham = hamnew - hamold
    #println("dH = $deltaham")
    if rand01() < min(1, exp(-deltaham))
        accp = true
    else
        # Revert back lattice field
        deepcopy!(lattice, latticeold)
        accp = false
    end
    return accp
end

"""
    Perform HMC update for some number of accepted iterations. The number of accepted iterations
depend on the length of fs! If fs! is not given, it is assumed to be thermalization processed
so it will update for iterations given by hmcparam.thermalizationiter. However, if fs! is not empty,
it is assumed to be measurements run and will update for iterations given by hmcparam.measurements

    Note that fs! should be a list of callback functions that accept lattice::Lattice as the sole input
argument. For each accepted update in the measurement runs, each callback function in fs! will be called
once to perform presumably some measurements on the current lattice.
"""
function HMCWilson_continuous_update!(lattice::Lattice, hmcparam::HMCParam, fs!...)
    # Determine if we are thermalizing lattice or making measurements
    thermalrun = false
    if length(fs!) == 0
        thermalrun = true
    end

    print_sep()
    if thermalrun
        hmciter = hmcparam.thermalizationiter
        println("--> Begin Thermalization: total accepted updates = $(hmciter)")
    else
        hmciter = hmcparam.measurements
        println("--> Begin Measurements: total measurements = $(hmcparam.measurements)")
    end
    print_sep()
    accptot = 0
    itertot = 0
    while accptot != hmciter
        itertot += 1
        accp = HMCWilson_update!(lattice, hmcparam)
        accptot += accp
        accprate = accptot/itertot
        if thermalrun
            println((@sprintf "Thermalization iterations: %4d (%4d/%4d completed" itertot accptot hmciter)*
                    (@sprintf ", accp rate = %.2f)" accprate))
        else
            println((@sprintf "Measurement iterations: %4d (%4d/%4d completed" itertot accptot hmciter)*
            (@sprintf ", accp rate = %.2f)" accprate))
            if accp
                # Call each function in fs! to do measurements
                for f! in fs!
                    f!(lattice)
                end
            end
        end
    end
    print_sep()
    if thermalrun
        println("--> Thermalization completed.")
    else
        println("--> Measurements completed.")
    end
    print_sep()
end

function test_HMC()
    # Lattice param
    nx = 32
    nt = 32
    kappa = 0.26 # Hopping parameter
    mass = (kappa^-1 - 4)/2
    beta = 2.5
    quenched = true

    # HMC param
    tau = 3
    integrationsteps = 200
    hmciter = 10000
    thermalizationiter = 1000
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
    accptot_hmc = 0
    for ihmciter in 1:hmcparam.niter
        println("Measurement steps: $ihmciter/$(hmcparam.niter)")
        accp = HMCWilson_update!(lattice, hmcparam)
        accptot += accp
        accptot_hmc += accp
        accprate = accptot/(ihmciter + hmcparam.thermalizationiter)
        if accp == 1
            plaq = measure_wilsonloop(lattice)
            plaqsum += plaq
            println("Accept = $accp; Acceptance rate = $accprate; Plaquette = $plaq")
            println("Avg plaq: $(plaqsum/accptot_hmc)")
        end
    end
end
#test_HMC()
