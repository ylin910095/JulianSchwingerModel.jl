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

function domaindecomp1()
    nx = 36
    nt = 60
    mass = 0.02
    beta = 10.0
    quenched = true
    thermaliter = 5
    measurements = 1
    tau = 1
    nsteps = 400
    check_herm = true # Do we want to check hermiticity of Q?

    # Domains for domain decomposition
    lambda0 = 1:2*10*nx
    lambda1 = 2*10*nx+1:2*50*nx # Frozen region
    lambda2 = 2*50*nx+1:2*nx*nt
    omega0star = 1:2*50*nx
    omega1star = 2*10*nx+1:2*nx*nt

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
        #A = matrix_decomp(Q_omg0s_inv, 1:size(Q_omg0s_inv)[1], lambda0)
        C = matrix_decomp(Q_omg1s_inv, length(lambda1)+1:size(Q_omg1s_inv)[1], 
                                       1:size(Q_omg1s_inv)[2])

        # Now compute the correction term
        M1 = matrix_decomp(Q_omg0s_inv, lambda0, 
                           1:size(Q_omg0s_inv)[2])
        M2 = matrix_decomp(Q, omega0star, lambda2)
        M3 = matrix_decomp(Q_omg1s_inv, 
                           1+length(lambda1):size(Q_omg1s_inv)[2], 
                           1:size(Q_omg1s_inv)[2])
        M4 = matrix_decomp(Q, omega1star, lambda0)
        T = M1*M2*M3*M4
        println(size(M1), " ", size(M2), " ", size(M3), " ", size(M4), " ")
        corr = inv(I - T)
        #corr = I + T + T^2 + T^3 + T^4 + T^5 + T^6 + T^7 + T^8

        # Put everything together
        approxQinv = - C * Q_lambda10 * corr * A

        # Now comparing the Qinv with approxQinv 
        Q_lambda20_inv = matrix_decomp(Qinv, lambda2, lambda0)
        Qout = approxQinv
        #println(sum(abs(Q_lambda20_inv - approxQinv))/length(approxQinv))
        #error()
        tol = 1e-15
        for i in 1:length(Q_lambda20_inv)
            if abs(real(Q_lambda20_inv[i] - approxQinv[i])) > tol || 
               abs(imag(Q_lambda20_inv[i] - approxQinv[i])) > tol
                #println(Q_lambda20_inv[i] - approxQinv[i])
            end
            dQinvout[i] += abs(Q_lambda20_inv[i] - approxQinv[i])/length(Q_lambda20_inv)
        end
    end

    #############################################################
    # End of definitions, start doing actual work
    #############################################################
    
    lattice = Lattice(nx, nt, mass, beta, quenched)
    hmcparam = HMCParam(tau, nsteps, thermaliter, measurements)
    print_lattice(lattice)

    # Finally do all measurements
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
    #npzwrite("dQinvout_l3636m0.02b10quenched.npz", dQinvout)
    #npzwrite("Qinvout_l3636m0.02b10quenched.npz", Qout)
    
end 

function domaindecomp2()

    # Setup lattice
    nx = 36
    nt = 100
    mass = 0.27
    beta = 10.0
    quenched = true
    thermaliter = 1000 # We load from existing lattice
    measurements = 50000
    tau = 1
    nsteps = 600

    # Setup domains
    lambda0 = 1:2*40*nx
    lambda1 = 2*40*nx+1:2*60*nx # Frozen region
    lambda2 = 2*60*nx+1:2*nx*nt
    omega0star = 1:2*60*nx
    omega1star = 2*40*nx+1:2*nx*nt

    # Sources for measurements and sources
    t0 = 1
    wallsource1 = FlatField(undef, Int(2*nx*nt))
    wallsource2 = FlatField(undef, Int(2*nx*nt))
    for i in 1:Int(nx*nt)
        if lin2corr(i, nx)[2] == t0
            wallsource1[dirac_comp1(i)] = 1.0
            wallsource1[dirac_comp2(i)] = 0.0
            wallsource2[dirac_comp1(i)] = 0.0
            wallsource2[dirac_comp2(i)] = 1.0
        else
            wallsource1[dirac_comp1(i)] = 0.0
            wallsource1[dirac_comp2(i)] = 0.0
            wallsource2[dirac_comp1(i)] = 0.0
            wallsource2[dirac_comp2(i)] = 0.0
        end
    end

    #--------- Start Working ---------#
    #lattice = load_lattice("./gauge/l$(nx)$(nt)q$(quenched)b$(beta)m$(mass).gauge")
    lattice = Lattice(nx, nt, mass, beta, quenched)
    hmcparam = HMCParam(tau, nsteps, thermaliter, measurements)
    print_lattice(lattice)

    # Define measurement routine
    a0corrlist = []
    g1corrlist = []
    pioncorrlist = []
    function _measure_all!(lattice::Lattice)
        Q = gamma5_Dslash_linearmap(lattice, lattice.mass)
        prop1 = minres_Q(Q, lattice, mass, gamma5mul(wallsource1))
        prop2 = minres_Q(Q, lattice, mass, gamma5mul(wallsource2))
        prop = [prop1, prop2]
        pioncorr = measure_pion(prop, lattice)
        a0corr = measure_a0(prop, lattice)
        g1corr = measure_g1(prop, lattice)
        println("pion[1:5]: ", real(pioncorr[1:5]))
        println("a0[1:5]:   ", real(a0corr[1:5]))
        println("g1[1:5]:   ", real(g1corr[1:5]))
        push!(pioncorrlist, pioncorr)
        push!(g1corrlist, g1corr)
        push!(a0corrlist, a0corr)
    end
    
    # Rethermalize the lattice
    HMCWilson_continuous_update!(lattice, hmcparam)

    # Do measurements
    HMCWilson_continuous_update!(lattice, hmcparam, _measure_all!)

    # Write to file
    pioncorrs = Array{Float64}(undef, hmcparam.measurements, nt)
    a0corrs =  Array{Float64}(undef, hmcparam.measurements, nt)
    g1corrs =  Array{Float64}(undef, hmcparam.measurements, nt)
    for ic in 1:hmcparam.measurements
        a0corrs[ic,:] = real(a0corrlist[ic] ./ nx)
        pioncorrs[ic,:] = real(pioncorrlist[ic] ./ nx)
        g1corrs[ic,:] = real(g1corrlist[ic] ./ nx)
    end
    npzwrite("a0_correlators_l$(nx)$(nt)q$(quenched)b$(beta)m$(mass).npz", a0corrs)
    npzwrite("pion_correlators_l$(nx)$(nt)q$(quenched)b$(beta)m$(mass).npz", pioncorrs)
    npzwrite("g1_correlators_l$(nx)$(nt)q$(quenched)b$(beta)m$(mass).npz", g1corrs)
end
#domaindecomp2()
domaindecomp1()