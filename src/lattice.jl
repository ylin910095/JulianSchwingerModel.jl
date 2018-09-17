
# Strcuture of Arrays
mutable struct Lattice
    # Inputs
    nx::Int64
    nt::Int64

    # Initilized(derived) variables
    ntot::Int64
    leftx::Array{Int64, 2}
    rightx::Array{Int64, 2}
    upt::Array{Int64, 2}
    downt::Array{Int64, 2}
    corr_indx::Array{Tuple{Int64, Int64}, 2} # coordinate tuple in the order of (x,t)

    # Inner constructor method to initilize structure
    function Lattice(nx::Int64, nt::Int64)
        ntot = nx * nt
        leftx  = Array{Int64, 2}(undef, nx, nt)
        rightx = Array{Int64, 2}(undef, nx, nt)
        upt = Array{Int64, 2}(undef, nx, nt)
        downt = Array{Int64, 2}(undef, nx, nt)
        corr_indx = Array{Tuple{Int64, Int64}, 2}(undef, nx, nt)
        for i in 1:ntot
            # Find the closest neighbors and coordinate indices
            leftx[i] = i - 1 
            rightx[i] = i + 1
            upt[i] = i + nt
            downt[i] = i - nt 
            corr_indx[i] = ((i-1)%nx + 1, 
                            floor((i-1)/nx) + 1)
            # Imposing boundary condition (periodic)
            for il in [leftx, rightx, upt, downt]
                if il[i] < 1
                    il[i] = ntot + il[i]
                elseif il[i] > ntot && il[i] < 2*ntot + 1
                    il[i] = il[i] - ntot
                end
            end
        end
        # The ordering here corresponds to the ordering of definition 
        # at the beginning of struct. If wrongly ordered, it will raise
        # some weird errors
        new(nx, nt, ntot, leftx, rightx, upt, downt, corr_indx)
    end
end

function test(nx::Int64, nt::Int64)
    # Testing routines
    @time lattice = Lattice(2^1, 2^1)
    println(lattice.ntot, " ", lattice.nx, " ", lattice.nt)
    println("left_x:  ", lattice.leftx)
    println("right_x: ", lattice.rightx)
    println("up_t:    ", lattice.upt)
    println("down_t:  ", lattice.downt)
    println("Coordinate indices: ", lattice.corr_indx)
end
