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
"""
psibar gamma5 psi correlators with wall source
"""
function pion_correlators()
    # Gauge parameters
    nx = 32
    nt = 32
    kappa = 0.26 # Hopping parameter = (2m0 + 4r)^{-1} at 2D where r is Wilson parameter
    mass = (kappa^-1 - 4)/2
    beta = 2.29
    quenched = false
    t0 = 0 # Location of wallsource
    saving_directory = "/home/ylin/scratch/schwinger_julia/src/gauge/"
    saving_prefix = @sprintf "%sl%d%db%.4fk%.4fseed%d-"

    allgaugeconfig = []
    #  Read all gauge files into a list
    for ifile in readdir(saving_directory)
        if isfile(saving_directory * ifile)
            push!(allgaugeconfig, isfile)
    end

    # Now invert propagators with wall source
    wallsource = Field(undef, Int(nx*nt))
    for i in 1:Int(nx*nt)
        if lin2corr(i)[2] == t0 
            wallsource[i] = [1.0, 1.0]
        end
    end

    pioncorrs = Vector{Vector{ComplexF64}}(undef, length(allgaugeconfig))
    for ifile in length(allgaugeconfig)
        lattice = load_lattice(ifile)
        # Use wallsource as initial guess too
        gy = cg_Q(lattice, mass, wallsouce, wallsource)
        anst = Vector{ComplexF64}(undef, lattice.nt) # Should all initilized to zero
        for i in 1:lattice.ntot
            anst[lattice.corr_indx[2]] += (conj(gy[i][1])*gy[i][1] +
                                           conj(gy[i][2])*gy[i][2])
        end
        push!(prioncorrs, pioncorr ./ nx ./ nx) 
end
pion_correlators()