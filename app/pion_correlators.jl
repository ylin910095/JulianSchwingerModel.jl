using Printf
using NPZ
using Profile

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

"""
psibar gamma5 psi correlators with wall source
"""
function meson_correlators()
    # Gauge parameters
    nx = 32
    nt = 32
    # Hopping parameter = (2m0 + 4r)^{-1} at 2D where r is Wilson parameter
    kappa = 0.22
    mass = (kappa^-1 - 4)/2
    beta = 6.0
    quenched = false
    t0 = 1 # Location of wallsource
    saving_directory = "./gauge/"

    allgaugeconfig = []
    #  Read all gauge files into a list
    for ifile in readdir(saving_directory)
        if isfile(saving_directory * ifile)
            push!(allgaugeconfig, ifile)
        end
    end

    # Now invert propagators with wall sources for each
    # Dirac component
    wallsource1 = Field(undef, Int(nx*nt))
    wallsource2 = Field(undef, Int(nx*nt))
    for i in 1:Int(nx*nt)
        if lin2corr(i, nx)[2] == t0
            wallsource1[i] = [1.0, 0.0]
            wallsource2[i] = [0.0, 1.0]
        else
            wallsource1[i] = [0.0, 0.0]
            wallsource2[i] = [0.0, 0.0]
        end
    end

    pioncorrs = Array{Float64}(undef, length(allgaugeconfig), nt)
    for (ic, ifile) in enumerate(allgaugeconfig)
        println("Current gauge: $ifile ($ic/$(length(allgaugeconfig)))")
        lattice = load_lattice(saving_directory * ifile)
        # Invert propagators all for both Dirac componenets
        prop1 = cg_Q(lattice, mass, wallsource1, gamma5mul(wallsource1))
        prop2 = cg_Q(lattice, mass, wallsource2, gamma5mul(wallsource2))
        prop = [prop1, prop2]
        # Tieups
        corr = measure_a0(prop, lattice)
        println(real(corr[1:10]))
        for it in 1:nt
            pioncorrs[ic, it] =  real((corr ./ nx)[it])
        end
    end
    npzwrite("a0_correlators.npz", pioncorrs)
end

"""
Perform both HMC updates and measurements on various meson correlators
"""
function meson_correlators_hmc()
    # Gauge parameters
    nx = 32
    nt = 32
    # Hopping parameter = (2m0 + 4r)^{-1} at 2D where r is Wilson parameter
    kappa = 0.22
    mass = (kappa^-1 - 4)/2
    beta = 6.0
    quenched = false
    t0 = 1 # Location of wallsource

    # HMC paramters
    thermalizationiter = 10
    measurements = 0
    integrationsteps = 10
    tau = 1

    # Now invert propagators with wall sources for each
    # Dirac component
    wallsource1 = Field(undef, Int(nx*nt))
    wallsource2 = Field(undef, Int(nx*nt))
    for i in 1:Int(nx*nt)
        if lin2corr(i, nx)[2] == t0
            wallsource1[i] = [1.0, 0.0]
            wallsource2[i] = [0.0, 1.0]
        else
            wallsource1[i] = [0.0, 0.0]
            wallsource2[i] = [0.0, 0.0]
        end
    end

    # Start thermalization
    lattice = Lattice(nx, nt, mass, beta, quenched)
    hmcparam = HMCParam(tau, integrationsteps, thermalizationiter, measurements)
    accptot = 0.0
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
    nomeas = 0 # Number of accepted measurements
    ihmciter = 0 # Total number of hmc iterations
    pioncorrs = Array{Float64}(undef, hmcparam.measurements, nt)
    a0corrs =  Array{Float64}(undef, hmcparam.measurements, nt)
    g1corrs =  Array{Float64}(undef, hmcparam.measurements, nt)
    while nomeas < hmcparam.measurements
        ihmciter += 1
        # First update the lattice
        println("Measurement steps: (hmciters: $ihmciter, measurements: $nomeas/$(
                 hmcparam.measurements))")
        accp = HMCWilson_update!(lattice, hmcparam)
        flush(stdout)
        accptot += accp
        accprate = accptot/(ihmciter + hmcparam.thermalizationiter)
        println("Accept = $accp; Acceptance rate = $accprate")
        if accp == 1
            # Invert propagators all for both Dirac componenets
            nomeas += 1
            prop1 = cg_Q(lattice, lattice.mass, wallsource1, gamma5mul(wallsource1))
            prop2 = cg_Q(lattice, lattice.mass, wallsource2, gamma5mul(wallsource2))
            prop = [prop1, prop2]
            # Tieups
            pioncorr = measure_pion(prop, lattice)
            a0corr = measure_a0(prop, lattice)
            g1corr = measure_g1(prop, lattice)
            println("pion[1:5]: ", real(pioncorr[1:5]))
            println("a0[1:5]:   ", real(a0corr[1:5]))
            println("g1[1:5]:   ", real(g1corr[1:5]))
            for it in 1:nt
                a0corrs[nomeas, it] =  real((a0corr ./ nx)[it])
                pioncorrs[nomeas, it] = real((pioncorr) ./ nx)[it]
                g1corrs[nomeas, it] = real((g1corr) ./ nx)[it]
            end
        end
    npzwrite("a0_correlators_l$(nx)$(nt)q$(quenched)b$(beta)k$(kappa).npz", a0corrs)
    npzwrite("pion_correlators_l$(nx)$(nt)q$(quenched)b$(beta)k$(kappa).npz", pioncorrs)
    npzwrite("g1_correlators_l$(nx)$(nt)q$(quenched)b$(beta)k$(kappa).npz", g1corrs)
    end
end

# Do some profiling
Profile.clear()
Profile.init(Int(1e10), 0.001)
@profile meson_correlators_hmc()
