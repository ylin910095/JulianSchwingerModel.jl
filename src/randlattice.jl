using Random

"""
Set seed for all random number generator 
"""
# Set the seeds for all prng generation
if ((@isdefined rngseed) && (@isdefined rng)) == false
    const global rngseed = 1234 
    const global rng = MersenneTwister(rngseed)
end

"""
Gaussian random number with mean of 0 and std of 1
"""
gauss() = randn(rng)
