using Printf

include("../src/lattice.jl")
include("../src/spinor.jl")
include("../src/dirac.jl")
include("../src/solvers.jl")
include("../src/randlattice.jl")
include("../src/leapfrog.jl")
include("../src/hmc_types.jl")
include("../src/measurements.jl")
include("../src/hmc.jl")
include("../src/io.jl")

function generating_gauge()
    # Gauge parameters
    nx = 32
    nt = 32
    kappa = 0.235 # Hopping parameter = (2m0 + 4r)^{-1} at 2D where r is Wilson parameter
    mass = (kappa^-1 - 4)/2
    beta = 10.0
    quenched = false
    saving_directory = "/home/ylin/Dropbox/julianschwinger/app/gauge/"

    # HMC parameters
    tau = 1
    integrationsteps = 300
    thermalizationiter = 500
    measurements = 10000

    # Create lattice
    lattice = Lattice(nx, nt, mass, beta, quenched)
    hmcparam = HMCParam(tau, integrationsteps, thermalizationiter, measurements)
    saving_prefix = @sprintf "%sl%d%db%.4fk%.4fseed%d-" saving_directory nx nt beta kappa rngseed
    accptot = 0

    # Thermalization
    print_lattice(lattice)
    for ithiter in 1:hmcparam.thermalizationiter
        println("Thermalization steps: $ithiter/$(hmcparam.thermalizationiter)")
        accp = HMCWilson_update!(lattice, hmcparam)
        accptot += accp
        accprate = accptot/ithiter
        plaq = measure_wilsonloop(lattice)
        println("Accept = $accp; Acceptance rate = $accprate; Plaquette = $plaq")
        flush(stdout)
    end
    # Actual measurements
    plaqsum = 0.0
    accptot_meas = 0
    nomeas = 0 # Number of accepted measurements
    ihmciter = 0 # Total number of hmc iterations
    while nomeas < hmcparam.measurements
        ihmciter += 1
        println("Measurement steps: (hmciters: $ihmciter, measurements: $nomeas/$(
                 hmcparam.measurements))")
        accp = HMCWilson_update!(lattice, hmcparam)
        flush(stdout)
        accptot += accp
        accptot_meas += accp
        accprate = accptot/(ihmciter + hmcparam.thermalizationiter)

        if accp == 1
            # Measure plaquette
            nomeas += 1
            plaq = measure_wilsonloop(lattice)
            plaqsum += plaq
            println("Accept = $accp; Acceptance rate = $accprate; Plaquette = $plaq")
            println("Avg plaq: $(plaqsum/accptot_meas)")

            # Save lattice
            paddedsuffix = lpad(accptot_meas, 6 ,"0")
            savename = @sprintf "%s%s.gauge" saving_prefix paddedsuffix
            save_lattice(lattice, savename)
            println("Saved gauge file: $savename")
        end
    end
end
generating_gauge()
