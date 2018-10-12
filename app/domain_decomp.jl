using NPZ
using LinearAlgebra
include("../src/include_all.jl")

"""
Given matrix A of linear transofrmation on some vector space
V indices by 1:length(A), a subspace Y (specificied by input of
type UnitRange which is a subset of 1:length(A)), and a subspace X of V,
return the block matrix B of A of dimensions length(X) by length(Y)
that is B: X -> Y.

Example:
    A = reshape(collect(1:15), (5,3))
    X = 1:2
    Y = 2:4
    B = matrix_decomp(A, Y, X) # Block matrix
"""
function matrix_decomp(A::AbstractArray, Y::UnitRange{Int64},
                                       X::UnitRange{Int64})
    # Safety Check
    if length(size(A)) != 2
        error("A must be 2D (length(size(A)) = $(length(size(A)))")
    elseif length(X) > size(A)[2] || length(Y) > size(A)[2]
        error("Invalid length of X or Y")
    end
    B = Array{eltype(A)}(undef, length(Y), length(X))
    # Note to put it into Coulomn-major loop
    for (counti, i) in enumerate(X)
        for (countj, j) in enumerate(Y)
            B[countj, counti] = A[j, i]
        end
    end
    return B
end

function main()
    nx = 36
    nt = 100
    mass = 0.02
    beta = 10.0
    quenched = true
    thermaliter = 10
    measurements = 1
    tau = 1
    nsteps = 400
    check_herm = true # Do we want to check hermiticity of Q?

    # Domains for domain decomposition
    lambda0 = 1:2*40*nx
    lambda1 = 2*40*nx+1:2*60*nx # Frozen region
    lambda2 = 2*60*nx+1:2*nx*nt
    omega0star = 1:2*60*nx
    omega1star = 2*40*nx+1:2*nx*nt

    # Define callback function for that measures absolute difference 
    # between exact and approximate propagators 
    dQinv = zero(Array{Float64}(undef, length(lambda1), length(lambda0)))

    function _approxQinv!(lattice::Lattice, dQinvout::Array{Float64}, Qout::Array{ComplexF64})
        # First construct explicit matrix form for gamma5_Dslash
        Q = zero(Array{ComplexF64}(undef, 2*lattice.ntot, 2*lattice.ntot))
        x = zero(FlatField(undef, 2*lattice.ntot))
        for i in 1:2*lattice.ntot
            x[i] = 1.0 
            Q[:, i] = gamma5_Dslash_wilson_vector(x, lattice, lattice.mass)
            x[i] = 0.0 
        end 
    
        # Check Hermiticity
        dQ = adjoint(Q) - Q
        if dQ == zero(Array{ComplexF64}(undef, 2*lattice.ntot, 2*lattice.ntot))
            println("--> Q hermiticity passed")
        else
            error("--> Q is not hermitian!")
        end
    
        # Find inverse
        Qinv = inv(Q)
    
        println("Decomposing Q...")
        Q_omg0s = matrix_decomp(Q, omega0star, omega0star)
        Q_omg1s = matrix_decomp(Q, omega1star, omega1star)
        Q_lambda10 = matrix_decomp(Q, omega1star, lambda0)
        Q_omg1s_inv = inv(Q_omg1s) 
        Q_omg0s_inv = inv(Q_omg0s)
    
        # Trim the matrix so we can multiply properly
        # the final matrix will be linear transformation 
        # from lambda0 to lambda2
        A = matrix_decomp(Q_omg0s_inv, lambda0, lambda0)
        C = matrix_decomp(Q_omg1s_inv, length(lambda1)+1:size(Q_omg1s_inv)[1], 
                                       1:size(Q_omg1s_inv)[2])

        # Now compute the correction term
        L1 = matrix_decomp(Q_omg0s_inv, lambda0, 
                           1:size(Q_omg0s_inv)[2])
        L2 = matrix_decomp(Q, omega0star, omega1star)
        L3 = matrix_decomp(Q_omg1s_inv, 1:size(Q_omg1s_inv)[1], 
                           1:size(Q_omg1s_inv)[2])
        L4 = matrix_decomp(Q, omega1star, lambda0)
        println(size(L1), " ", size(L2), " ", size(L3), " ", size(L4), " ")
        corr = inv(I - L1*L2*L3*L4)
        #corr = I 

        # Put everything together
        approxQinv = - C * Q_lambda10 * corr * A

        # Now comparing the Qinv with approxQinv 
        Q_lambda20_inv = matrix_decomp(Qinv, lambda2, lambda0)
        Qout = approxQinv
        #println(sum(Q_lambda20_inv - approxQinv))
        #error()
        tol = 1e-15
        for i in 1:length(Q_lambda20_inv)
            if abs(real(Q_lambda20_inv[i] - approxQinv[i])) > tol || 
               abs(imag(Q_lambda20_inv[i] - approxQinv[i])) > tol
                println(Q_lambda20_inv[i] - approxQinv[i])
            end
            dQinvout[i] += abs(Q_lambda20_inv[i] - approxQinv[i])
        end
    end

    #############################################################
    # End of definitions, start doing actual work
    #############################################################

    lattice = Lattice(nx, nt, mass, beta, quenched)
    hmcparam = HMCParam(tau, nsteps, thermaliter, measurements)
    print_lattice(lattice)

    # Thermalize the lattice
    HMCWilson_continuous_update!(lattice, hmcparam)

    # Do measurements with given measurement routine
    Qout = zero(Array{ComplexF64}(undef, length(lambda2), length(lambda0)))
    dQinvout = zero(Array{Float64}(undef, length(lambda2), length(lambda0)))
    f(lattice::Lattice) = _approxQinv!(lattice, dQinvout, Qout)
    HMCWilson_continuous_update!(lattice, hmcparam, f)

    # Save to npz to be read by python notebook
    dQinvout = dQinvout ./ hmcparam.measurements
    Qout = Qout ./ hmcparam.measurements
    println("sum dQinvout = $(sum(dQinvout))")
    npzwrite("dQinvout_l3636m0.02b10quenched.npz", dQinvout)
    npzwrite("Qinvout_l3636m0.02b10quenched.npz", Qout)
end 
main()