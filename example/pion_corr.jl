# Example for how to make pion two-point correlator from scratch

using JulianSchwingerModel
import TensorOperations, NPZ


# Set up lattice parameters
nx = 32
nt = 32
kappa = 0.26                # hopping parameter
mass = (kappa^-1 - 4)/2
beta = 2.5
quenched = true

# Set up HMC parameters
tau = 1                     # leapfrog integration time
integrationsteps = 300
thermalizationiter = 10     # total number of accepted thermalization
hmciter = 10                # total number of accepted measurements

# Initilize lattice
lattice = Lattice(nx, nt, mass, beta, quenched)

# Put all HMC parameters in a type to be passed around
hmcparam = HMCParam(tau, integrationsteps, thermalizationiter, hmciter)

# Print basic lattice information
print_lattice(lattice)

# Thermalize lattice
HMCWilson_continuous_update!(lattice, hmcparam)

# Make some propagator wall sources
# zero will make sure all entries are initilized to zero inplace
t0 = 1 # which timeslice to put the sources in
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
    prop = [minres_Q(Q, lattice, lattice.mass, gamma5mul(wallsource1)), 
            minres_Q(Q, lattice, lattice.mass, gamma5mul(wallsource2))]

    # Now we want to tieup the propagators to measure pion correlators.
    # To do this, we will use TensorOperations 
    # (see https://github.com/Jutho/TensorOperations.jl) to simplify
    # the indices contraction. However, to use this package, we have to convert 
    # prop list into an array of shape (2, 2, lattice.nx, lattice.nt),
    # where the number 2 represents two possible dirac indices for the source. 
    # There is no site indices for the source as we are using wall sources;
    # the second number, 2, is the sink dirac index, and lattice.nx and lattice.nt
    # are the sink site coordinates.
    projectfield = Array{ComplexF64}(undef, 2, 2, lattice.ntot)
    tensorprop1 = reshape(prop[1], 2, lattice.ntot)
    tensorprop2 = reshape(prop[2], 2, lattice.ntot)
    projectfield[1, :, :] = tensorprop1
    projectfield[2, :, :] = tensorprop2
    projectfield = reshape(projectfield, (2, 2, lattice.nx, lattice.nt))

    # Now we can perform the Wick contractions easily for pions
    D = Array{ComplexF64}(undef, lattice.nt, lattice.nt)
    TensorOperations.@tensor begin
        D[t1, t2] = conj(projectfield[a, i, b, t1]) * projectfield[a, i, b, t2]
    end
    ans = [real(D[t, t]) for t in 1:lattice.nt]
    println("--> pion corr[1:10]: ", ans[1:10])

    # Append to output array
    push!(corrdata, ans)
end

# Now we can do the measurements
pioncorr = [] # output correlator holder
measfunc(lattice::Lattice) = _pioncorr!(pioncorr, lattice)

# Note that we use the same function to do the measurements.
# However, if we provide it with callback function(s), it will
# assume to be measurement run and call all of those functions 
# individually
HMCWilson_continuous_update!(lattice, hmcparam, measfunc)

# Now save the correlators into npz format 
# need to convert to array first or npzwrite will fail
outarray = Array{Float64}(undef, hmciter, nt)
for i in 1:length(pioncorr)
    outarray[i, :] = pioncorr[i]
end
npzfn = "pioncorr_l$(nx)$(nt)q$(quenched)b$(beta)m$(mass).npz"
NPZ.npzwrite("pioncorr_l$(nx)$(nt)q$(quenched)b$(beta)m$(mass).npz", outarray)
println("--> Saved to $npzfn")
