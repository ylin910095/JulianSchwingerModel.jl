"""
Set seed for all random number generators
"""
# Set the seeds for all prng generation
if ((@isdefined rngseed) && (@isdefined rng)) == false
    const global rngseed = 8888
    const global rng = Random.MersenneTwister(rngseed)
end

"""
Gaussian random number with mean of 0 and std of 1
"""
gauss() = Random.randn(rng)

"""
Random number between 0 and 1
"""
rand01() = Random.rand(rng, Float64)
