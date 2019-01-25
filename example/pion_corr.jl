# Example for how to make pion two-point correlator from scratch

using JulianSchwingerModel
import TensorOperations, NPZ, Printf, Statistics, Glob

"""
# Set up lattice parameters
nx = 32
nt = 32
kappa = 0.26                # hopping parameter
mass = (kappa^-1 - 4)/2
beta = 2.5
quenched = true
"""


# Lattice info
DEBUG = true
nx = 32
nt = 32
kappa = 0.2               # hopping parameter
#kappa = 0.15
mass = (kappa^-1 - 4)/2
beta = 10.0
quenched = true
series = "b"
start_no = 1
end_no = 100
no_sub = 20 # number of sublattice files. sub_no = 0 means no sublattice files.


"""
# Set up lattice parameters
DEBUG = true
nx = 32
nt = 96
kappa = 0.24               # hopping parameter
#kappa = 0.15
mass = (kappa^-1 - 4)/2
beta = 6.0
quenched = true
"""

"""
nx = 8
nt = 8
kappa = 0.2               # hopping parameter
mass = (kappa^-1 - 4)/2
beta = 10.0
quenched = true
"""

# "hmc" or "metropolis" for gauge field evolution
gauge = "metropolis"
thermalizationiter = 200     # total number of thermalization
#load_folder = "../gauge_metropolis"
load_folder = nothing

# Set up specific gauge evolution parameters
if gauge == "hmc"
    tau = 6                     # leapfrog integration time
    integrationsteps = 400
elseif gauge == "metropolis"
    epsilon = 0.3
    nwait = 200
end

# Now load gague files if requested
if load_folder != nothing
    println("--> Loading gauge files from $load_folder...")
    n0 = end_no-start_no+1
    all_gaugefn = Array{String, 2}(undef, n0, no_sub+1) # +1 to include top level file
    lat_list = Array{Lattice, 2}(undef, n0, no_sub+1)
    prefix = prefix = @Printf.sprintf "%s/l%d%db%.4fk%.4f" load_folder nx nt beta kappa
    for in0 in start_no:end_no
        n0prefix = "$(prefix)_$(series)$(in0)"
        all_gaugefn[in0, 1] = "$(n0prefix).metro"
        lat_list[in0, 1] = load_lattice(all_gaugefn[in0, 1])
        for in1 in 1:no_sub
            n1fn = "$(n0prefix)_sub$(in1).metro"
            all_gaugefn[in0,1+in1] = n1fn
            lat_list[in0, 1+in1] = load_lattice(n1fn)
        end
    end
    println("--> All loaded. Total n0 = $n0, total n1 = $(no_sub)")
end

if load_folder == nothing
    measiter = 10              # total number of measurements
else
    measiter = length(lat_list)
end

if load_folder == nothing
    # Initilize lattice
    lattice = Lattice(nx, nt, mass, beta, quenched)

    if gauge == "hmc"
        # Put all HMC parameters in a type to be passed around
        hmcparam = HMCParam(tau, integrationsteps, thermalizationiter, measiter)
    end

    # Print basic lattice information
    print_lattice(lattice)

    # Only hmc and metropolis are implemented for gauge evolution
    @assert (gauge == "hmc" || gauge == "metropolis") "Unknown keyword for gauge: $(gauge)!"

    # Thermalize lattice
    if gauge == "hmc"
        HMCWilson_continuous_update!(lattice, hmcparam)
    elseif gauge == "metropolis"
        for i in 1:thermalizationiter
            accprate = metropolis_update!(epsilon, lattice)
            println((Printf.@sprintf "Thermal iterations (metropolis): %4d/%4d completed" i thermalizationiter)*
                    (Printf.@sprintf ", current accp rate = %.5f" accprate))
        end
    end
else
    print_lattice(lat_list[1])
    lattice = lat_list[1] # place holder for source construction
end

# Make some propagator wall sources
# zero will make sure all entries are initilized to zero inplace
t0 = 6 # which timeslice to put the sources in
wallsource1 = zero(FlatField(undef, 2*Int(nx*nt))) # first dirac component
wallsource2 = zero(FlatField(undef, 2*Int(nx*nt))) # second dirac component

# Loop over all lattice sites
for i in 1:Int(nx*nt)
    # Convert linear index to coordinates (x, t)
    if lattice.corr_indx[i][2] == t0
        wallsource1[dirac_comp1(i)] = 1.0
        wallsource1[dirac_comp2(i)] = 0.0
        wallsource2[dirac_comp1(i)] = 0.0
        wallsource2[dirac_comp2(i)] = 1.0
    end
end
# Now we have to define a callback function for HMC to
# perform measurements. This function will be called in each
# accepted HMC iterations with sole input argument
# of lattice::Lattice, the current updated configuration
# Append measurements to corrdata list
function _pioncorr!(corrdata, lattice::Lattice)
    # Create linearmap type so it can be used by IterativeSolvers
    Q = gamma5_Dslash_linearmap(lattice, lattice.mass)

    # Invert propagators, we need to multiply by gamma5 because
    # Q := gamma5 * Dslash. Also match valence masses to sea masses
    ms1 = wallsource1
    ms2 = wallsource2
    """
    prop = [minres_Q(Q, lattice, lattice.mass, ms1),
            minres_Q(Q, lattice, lattice.mass, ms2)]
    """

    prop = [minres_Q(Q, lattice, lattice.mass, wallsource1),
            minres_Q(Q, lattice, lattice.mass, wallsource2)]



    # Now we want to tieup the propagators to measure pion correlators.
    # To do this, we will use TensorOperations
    # (see https://github.com/Jutho/TensorOperations.jl) to simplify
    # the indices contraction. However, to use this package, we have to convert
    # prop list into an array of shape (2, 2, lattice.nx, lattice.nt),
    # where the first number 2 represents two possible dirac indices for the source.
    # There is no site indices for the source as we are using wall sources;
    # the second number, 2, is the sink dirac index, and lattice.nx and lattice.nt
    # are the sink site coordinates.
    projectfield = zero(Array{ComplexF64}(undef, 2, 2, lattice.ntot))
    tensorprop1 = reshape(prop[1], 2, lattice.ntot)
    tensorprop2 = reshape(prop[2], 2, lattice.ntot)
    projectfield[1, :, :] = tensorprop1
    projectfield[2, :, :] = tensorprop2
    projectfield = reshape(projectfield, (2, 2, lattice.nx, lattice.nt))

    # Now we can perform the Wick contractions easily for pions
    D = zero(Array{ComplexF64}(undef, lattice.nt, lattice.nt))

    TensorOperations.@tensor begin
        D[t1, t2] = conj(projectfield[a, b, i, t1]) * projectfield[a, b, i, t2]
    end


    """
    TensorOperations.@tensor begin
        D[t1, t2] = gamma5[a, c] * conj(projectfield[g, c, x, t1]) *
                    gamma5[g, b] * projectfield[b, a, x, t2]
    end
    """

    ans = [real(D[t, t])/nx for t in t0:lattice.nt]
    println("--> pion corr ", ans[:])

    # Append to output array
    push!(corrdata, ans)
end

pioncorr = [] # output correlator holder
if load_folder == nothing
    # Now we can do the measurements
    measfunc(lattice::Lattice) = _pioncorr!(pioncorr, lattice)

    # Note that we use the same function to do the measurements.
    # However, if we provide it with callback function(s), it will
    # assume to be measurement run and call all of those functions
    # individually
    if gauge == "hmc"
        HMCWilson_continuous_update!(lattice, hmcparam, measfunc)
    elseif gauge == "metropolis"
        for i in 1:measiter
            for ii in 1:nwait
                accprate = metropolis_update!(epsilon, lattice)
                println((Printf.@sprintf "Measurement iterations (metropolis): %4d/%4d completed" i measiter)*
                        (Printf.@sprintf ", current accp rate = %.5f" accprate))
            end
            measfunc(lattice)
        end
    end
else
    # Now we can do the measurements
    for in0 in 1:n0
        bincorr = zero(Array{Float64, 2}(undef, no_sub+1, nt-t0+1)) # bin over all n1 configs
        for in1 in 1:no_sub+1
            lattice = lat_list[in0, in1]
            temp_list = [] # temporary place holder
            _pioncorr!(temp_list, lattice) # do measurements
            bincorr[in1, :] = temp_list[1]
        end

        # Bin the data
        pl = Statistics.mean(bincorr, dims=1)[1,:]
        push!(pioncorr, pl)
        println(Printf.@sprintf "Measurement iterations (loaded): %4d/%4d completed" in0 n0)
    end
end

# Now save the correlators into npz format
# need to convert to array first or npzwrite will fail
outarray = Array{Float64}(undef, length(pioncorr), nt-t0+1)
for i in 1:length(pioncorr)
    outarray[i, :] = pioncorr[i]
end
finavg = Statistics.mean(outarray, dims=1)
println("Final average: $finavg")
npzfn = "../data/pion_corr_l$(nx)$(nt)q$(quenched)b$(beta)m$(mass)t0$(t0).npz"
NPZ.npzwrite(npzfn, outarray)
println("--> Saved to $npzfn")
