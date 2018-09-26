using Printf

include("./lattice.jl")
include("./spinor.jl")
include("./dirac.jl")
include("./solvers.jl")
include("./randlattice.jl")
include("./leapfrog.jl")
include("./hmc_types.jl")
include("./measurements.jl")
include("./hmc.jl")
include("./io.jl")

function generating_gauge()
    # Gauge parameters
    nx = 32
    nt = 32
    kappa = 0.26 # Hopping parameter = (2m0 + 4r)^{-1} at 2D where r is Wilson parameter
    mass = (kappa^-1 - 4)/2
    beta = 2.29
    quenched = false
    saving_directory = "/home/ylin/scratch/schwinger_julia/src/gauge/"

    # HMC parameters
    tau = 1
    integrationsteps = 200
    hmciter = 10000
    thermalizationiter = 10

    # Create lattice
    lattice = load_lattice("/home/ylin/scratch/schwinger_julia/src/gauge/l3232b2.2900k0.2600seed1234-000224.gauge")
    hmcparam = HMCParam(tau, integrationsteps, hmciter, thermalizationiter)
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
    accptot_hmc = 0
    for ihmciter in 1:hmcparam.niter
        println("Measurement steps: $ihmciter/$(hmcparam.niter)")
        accp = HMCWilson_update!(lattice, hmcparam)
        flush(stdout)
        accptot += accp
        accptot_hmc += accp
        accprate = accptot/(ihmciter + hmcparam.thermalizationiter)

        if accp == 1
            # Measure plaquette
            plaq = measure_wilsonloop(lattice)
            plaqsum += plaq
            println("Accept = $accp; Acceptance rate = $accprate; Plaquette = $plaq")
            println("Avg plaq: $(plaqsum/accptot_hmc)")

            # Save lattice
            paddedsuffix = lpad(accptot_hmc, 6 ,"0")
            savename = @sprintf "%s%s.gauge" saving_prefix paddedsuffix
            save_lattice(lattice, savename)
            println("Saved gauge file: $savename")
        end
    end
end
generating_gauge()