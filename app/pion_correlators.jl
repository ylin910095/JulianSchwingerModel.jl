using Printf
using NPZ
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

"""
psibar gamma5 psi correlators with wall source
"""
function pion_correlators()
    # Gauge parameters
    nx = 32
    nt = 32
    # Hopping parameter = (2m0 + 4r)^{-1} at 2D where r is Wilson parameter
    kappa = 0.26
    mass = (kappa^-1 - 4)/2
    beta = 2.29
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

    # Now invert propagators with wall source
    wallsource = Field(undef, Int(nx*nt))
    for i in 1:Int(nx*nt)
        if lin2corr(i, nx)[2] == t0
            wallsource[i] = [1.0, 1.0]
        else
            wallsource[i] = [0.0, 0.0]
        end
    end

    pioncorrs = Array{Float64}(undef, length(allgaugeconfig), nt)
    for (ic, ifile) in enumerate(allgaugeconfig)
        anst = zero(Vector{Float64}(undef, nt)) # Should all initilized to zero
        println("Current gauge: $ifile ($ic/$(length(allgaugeconfig)))")
        lattice = load_lattice(saving_directory * ifile)
        # Use wallsource as initial guess too
        gy = cg_Q(lattice, mass, wallsource, wallsource)
        for i in 1:lattice.ntot
            anst[lattice.corr_indx[i][2]] += (conj(gy[i][1])*gy[i][1] +
                                              conj(gy[i][2])*gy[i][2])
        end
        for it in 1:nt
            pioncorrs[ic, it] =  real(anst ./ nx)[it]
        end
    end
    npzwrite("pion_correlators.npz", pioncorrs)
end
pion_correlators()
