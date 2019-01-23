using JulianSchwingerModel
using Printf, TensorOperations, Statistics, LinearAlgebra
using NPZ


# Lattice info
DEBUG = true
nx = 32
nt = 32
kappa = 0.2               # hopping parameter
#kappa = 0.15
mass = (kappa^-1 - 4)/2
beta = 10.0
quenched = true

load_folder = "../gauge_metropolis/" # nothing or the directory containing all gauge files

# Which meson do we want?
# operator = \bar{\psi} \Gamma \psi
Gamma = Matrix{Float64}(I, 2, 2) # must be a 2x2 array
#Gamma = gamma5
# Metropolis controls
nthermal = 200
iter_wait = 20 # how many iterations to wait before performing next measurements
n1_wait = 4
epsilon = 0.3

# Define domain according to
# https://arxiv.org/pdf/1812.01875.pdf 
# Note that lambda1 has two disconnected domain lambda11 and lambda12
x0 = 6


lambda0 = 1:12
lambda2 = 17:28
lambda11 = 13:16
lambda12 = 29:32

# this is for nt = 96
"""
lambda0 = 1:40
lambda2 = 49:88
lambda11 = 41:48
lambda12 = 89:96
"""

"""
lambda0 = 1:4
lambda2 = 18:23
lambda11 = 5:17
lambda12 = 24:32
"""

"""
lambda0 = 1:16
lambda2 = 33:48
lambda11 = 17:32
lambda12 = 49:64
"""

"""
nx = 8
nt = 8
kappa = 0.2               # hopping parameter
mass = (kappa^-1 - 4)/2
beta = 10.0
quenched = true

# Domains
lambda0 = 1:2
lambda2 = 5:6
lambda11 = 3:4
lambda12 = 7:8
"""


@assert (x0 in lambda0) == true "x0 must be in lambda0!"

# Now load gague files if requested
if load_folder != nothing
    lattice_list = []
    all_gaugefn = readdir(load_folder)
    println("--> Loading gauge files from $load_folder...")
    for ifn in all_gaugefn
        if endswith(ifn, "metro")
            push!(lattice_list,
                load_lattice("$(load_folder)$(ifn)"))
        end
    end
    println("--> All loaded. Total number = $(length(lattice_list))")
end 

# Multi-level update parameters
if load_folder == nothing
    n0 = 100 # upper level measurements (whole domain)
else
    n0 = length(lattice_list)
end
n1 = 10 # lower level measurements (omega0* and omega1*)

# Output data filename
t0 = minimum(lambda2) - x0
savname = "../data/a0_corr_domain_l$(nx)$(nt)q$(quenched)b$(beta)m$(mass)_t0$(t0).npz"

function decompose_domain(force=false)
    # Truncate to disconnected domains if NOT defined
    if (((@isdefined lat_lambda0) && (@isdefined lat_lambda2)) == false ||
        force == true)
        if DEBUG == true
            println("--> DEBUG: redefining lambda0, 2, 11, and 12")
        end
        global lat_lambda0 = trunc_lattice(latcopy, lambda0)
        global lat_lambda2 = trunc_lattice(latcopy, lambda2)
        global lat_lambda11 = trunc_lattice(latcopy, lambda11)
        global lat_lambda12 = trunc_lattice(latcopy, lambda12)

        # Enable safety check ONLY if we first truncate lattice
        # When we later update partial domain, say lambda0,
        # then the ghost cells will not be updated in lambda11
        # and lambda12 and sanity check will fail when we try to 
        # stack them.
        # TODO: automatically updating ghost cells of other domains 
        #       when adjacent domain is updated.
        safety_check = true 
    else
        safety_check = false
    end

    # Stack domains (or restack domains if defined)
    p = false # not periodic
    temp1 = stack_sublattice(lat_lambda12, lat_lambda0, p, safety_check)
    global lat_omega0s = stack_sublattice(temp1, lat_lambda11, p, safety_check)
    temp1 = stack_sublattice(lat_lambda11, lat_lambda2, p, safety_check)
    global lat_omega1s = stack_sublattice(temp1, lat_lambda12, p, safety_check)

    # Reconstruct whole lattice
    temp1 = stack_sublattice(lat_lambda0, lat_lambda11, p, safety_check)
    temp2 = stack_sublattice(temp1, lat_lambda2, p, safety_check)

    # Make sure periodic flag is true
    global latcopy = stack_sublattice(temp2, lat_lambda12, true, safety_check)
    @assert latcopy.boundary_cond == "periodic" "The whole lattice should be periodic!"
end

# Start doing work
lattice = Lattice(nx, nt, mass, beta, quenched, 
                  boundary_cond="periodic")


# Always thermalize first!
if load_folder == nothing
    print_lattice(lattice)
    print_sep()
    println("--> Begin Thermalization: total updates = $nthermal")
    print_sep()
    for i in 1:nthermal 
        accprate = metropolis_update!(epsilon, lattice)
        println((Printf.@sprintf "Measurement iterations: %4d/%4d completed" i nthermal)*
                (Printf.@sprintf ", current accp rate = %.5f)" accprate))
    end
else
    print_lattice(lattice_list[1])
end

# Container for final correlator
raw_corr = zero(Array{ComplexF64, 2}(undef, n0, length(lambda2)))

for imol in 1:n0
    print_sep()
    println("--> n0 updates of whole lattice: total updates = $n0")
    print_sep()
    if load_folder == nothing
        for ii in 1:iter_wait
            accprate = metropolis_update!(epsilon, lattice)
            println((Printf.@sprintf "Measurement iterations: %4d/%4d completed" imol n0)*
                    (Printf.@sprintf ", current accp rate = %.5f)" accprate))
        end
    else
        lattice = lattice_list[imol]
        println("--> Loaded gauge = $(all_gaugefn[imol])")
        println(Printf.@sprintf "Measurement iterations: %4d/%4d completed" imol n0)
    end

    # lattice variable will ONLY be updated with n0 update
    # whereas latcopy will be updated within n1 updates 
    # for ease of manipulation
    global latcopy = deepcopy(lattice) 
    # Force redefining regions
    decompose_domain(true)

    # Internal boundaries of lambda1 in lambda1
    lam1_1min = 1 
    lam1_1max = nx
    lam1_2min = lat_lambda11.ntot - nx + 1 
    lam1_2max = lat_lambda11.ntot
    lam1_3min = lat_lambda11.ntot + 1
    lam1_3max = lat_lambda11.ntot + nx
    lam1_4min = lat_lambda11.ntot + lat_lambda12.ntot - lat_lambda11.nx + 1
    lam1_4max = lat_lambda11.ntot + lat_lambda12.ntot
    lam1_t1 = lam1_1min : lam1_1max
    lam1_t2 = lam1_2min : lam1_2max
    lam1_t3 = lam1_3min : lam1_3max
    lam1_t4 = lam1_4min : lam1_4max

    # Internal boundaries of lambda1 in omega1s 
    lam1_omega1s_1min = 1 
    lam1_omega1s_1max = nx
    lam1_omega1s_2min = lat_lambda11.ntot -  lat_lambda11.nx + 1 
    lam1_omega1s_2max = lat_lambda11.ntot
    lam1_omega1s_3min = lat_lambda11.ntot + lat_lambda2.ntot + 1
    lam1_omega1s_3max = lat_lambda11.ntot + lat_lambda2.ntot + nx
    lam1_omega1s_4min = lat_omega1s.ntot - nx + 1 
    lam1_omega1s_4max = lat_omega1s.ntot
    lam1_omega1s_t1 = lam1_omega1s_1min : lam1_omega1s_1max
    lam1_omega1s_t2 = lam1_omega1s_2min : lam1_omega1s_2max
    lam1_omega1s_t3 = lam1_omega1s_3min : lam1_omega1s_3max
    lam1_omega1s_t4 = lam1_omega1s_4min : lam1_omega1s_4max

    # Interior boundaries of lambda0 in omega0s
    bt1 = lat_lambda12.ntot + 1
    bt2 = lat_lambda12.ntot + lat_lambda0.ntot - nx + 1
    lam0_omega0s_t1 = bt1:bt1+nx-1 
    lam0_omega0s_t2 = bt2:bt2+nx-1

    # Interior boundaries of lambda0 in the whole lattice
    lam0_whole_1min = 1
    lam0_whole_1max = nx
    lam0_whole_2min = lat_lambda0.ntot - nx + 1
    lam0_whole_2max = lat_lambda0.ntot
    lam0_whole_t1 = lam0_whole_1min : lam0_whole_1max
    lam0_whole_t2 = lam0_whole_2min : lam0_whole_2max

    # This phi is the same phi as defined in page 4 of
    # https://arxiv.org/abs/1812.01875
    # 4 because of 4 timeslices of lambda1 interior boundaries
    phi = zero(Array{ComplexF64, 4}(undef, n1, 2 , latcopy.ntot, 2))

    # Loop over n1 updates of omega0s region
    print_sep()
    println("--> n1 updates of lambda0: total updates = $n1")
    print_sep()
    for in1_omega0s in 1:n1
        for ii in 1:n1_wait
            accprate = metropolis_update!(epsilon, lat_lambda0)
            println((Printf.@sprintf "Measurement iterations: %4d/%4d completed" in1_omega0s n1)*
                    (Printf.@sprintf ", current accp rate = %.2f)" accprate))
        end

        # Decompose domain after each update
        decompose_domain()

        # Now make wallsource for omega0s domain
        wallsource1 = zero(FlatField(undef, 2*lat_omega0s.ntot)) # first dirac component
        wallsource2 = zero(FlatField(undef, 2*lat_omega0s.ntot)) # second dirac component

        # Loop over all lattice sites
        for i in 1:lat_omega0s.ntot
            # Convert linear index to coordinates (x, t)
            if lat_omega0s.corr_indx[i][2] == x0 + lat_lambda12.nt # because we stack them
                wallsource1[dirac_comp1(i)] = 1.0
                wallsource1[dirac_comp2(i)] = 0.0
                wallsource2[dirac_comp1(i)] = 0.0
                wallsource2[dirac_comp2(i)] = 1.0
            end
        end

        # Use the wallsource to invert propagator in omega0s domain
        Q_omega0s = gamma5_Dslash_linearmap(lat_omega0s, lat_omega0s.mass)
        prop = [minres_Q(Q_omega0s, lat_omega0s, 
                        lat_omega0s.mass, wallsource1),
                minres_Q(Q_omega0s, lat_omega0s, 
                        lat_omega0s.mass, wallsource2)]

        # Create matrix field as usual 
        Dinv_omega0s = zero(Array{ComplexF64}(undef, 2, lat_omega0s.ntot, 2))
        tensorprop1 = reshape(prop[1], (2, lat_omega0s.ntot))
        tensorprop2 = reshape(prop[2], (2, lat_omega0s.ntot))
        Dinv_omega0s[:, :, 1] = tensorprop1
        Dinv_omega0s[:, :, 2] = tensorprop2

        # Compute A = \sum_{z}Q_{\Lambda_{10}}(w, z) * Q^{-1}_{\Omega_0^*}(z, x)
        Dinv_omega0s_whole = zero(Array{ComplexF64}(undef, 2, latcopy.ntot, 2))


        # Need to be REALLY careful of how you fill in the blank here
        # since in omega0s, lambda12 comes first whereas it comes LAST
        # in the whole lattice. ONLY lambda0 values are retained, the rest
        # of values MUST BE ZEROS. If not set to zeros, it will again fail.
        Dinv_omega0s_whole[:,1:lat_lambda0.ntot,:] = Dinv_omega0s[:,lat_lambda12.ntot+1:lat_lambda12.ntot+lat_lambda0.ntot,:]
            
        # Flatten it so we can use linearmap
        Dinv_omega0s_whole = reshape(Dinv_omega0s_whole, 2*latcopy.ntot, 2)
        Qwhole = gamma5_Dslash_linearmap(lattice, lattice.mass)

        """
        Now we want to compare it to exact values
        """
        BU = Dinv_omega0s_whole
        exact = false
        
        if exact == true
            wallsource1 = zero(FlatField(undef, 2*nx*nt)) # first dirac component
            wallsource2 = zero(FlatField(undef, 2*nx*nt)) # second dirac component
        
            # Loop over all lattice sites
            for i in 1:lattice.ntot
                # Convert linear index to coordinates (x, t)
                if lattice.corr_indx[i][2] == x0 # because we stack them
                    wallsource1[dirac_comp1(i)] = 1.0
                    wallsource1[dirac_comp2(i)] = 0.0
                    wallsource2[dirac_comp1(i)] = 0.0
                    wallsource2[dirac_comp2(i)] = 1.0
                end
            end
            """
            print_sep()
            println("wallsource1, x0 = $(x0)")
            display(reshape(wallsource1, 2, nx, nt)[1,:,:])
            println()
            print_sep()
            println()
            """
        
            # Use the wallsource to invert propagator in omega0s domain
            Q = gamma5_Dslash_linearmap(lattice, lattice.mass)
            prop = [minres_Q(Q, lattice, 
                            lattice.mass, wallsource1),
                    minres_Q(Q, lattice, 
                            lattice.mass, wallsource2)]
        
            # Create matrix field as usual 
            Dinv_omega0s_whole = zero(Array{ComplexF64}(undef, 2*lattice.ntot, 2))
            #tensorprop1 = reshape(prop[1], (2, lattice.ntot))
            #tensorprop2 = reshape(prop[2], (2, lattice.ntot))
            Dinv_omega0s_whole[:, 1] = prop[1]
            Dinv_omega0s_whole[:, 2] = prop[2]
        
            # Truncate to lambda0
            for i in 1:lattice.ntot
                if (i in 1:lat_lambda0.ntot) == false
                    Dinv_omega0s_whole[dirac_comp1(i), :] = [0,0]
                    Dinv_omega0s_whole[dirac_comp2(i), :] = [0,0]
                end
            end

            # Redefine omega0s
            Dinv_omega0s = zero(Array{ComplexF64}(undef, 2, lat_omega0s.ntot, 2))
            temp1 = reshape(prop[1], (2,nx*nt))
            temp2 = reshape(prop[2], (2,nx*nt))
            Dinv_omega0s[:, 1:lat_lambda12.ntot,1] = temp1[:, latcopy.ntot-lat_lambda12.ntot+1:latcopy.ntot]
            Dinv_omega0s[:, 1:lat_lambda12.ntot,2] = temp2[:, latcopy.ntot-lat_lambda12.ntot+1:latcopy.ntot]
            Dinv_omega0s[:, lat_lambda12.ntot+1:lat_omega0s.ntot,1] = (
                temp1[:, 1:lat_lambda0.ntot+lat_lambda11.ntot]
            )
            Dinv_omega0s[:, lat_lambda12.ntot+1:lat_omega0s.ntot, 2] = (
                temp2[:, 1:lat_lambda0.ntot+lat_lambda11.ntot]
            )
        end # exact == true
        
        """
        # If this comparison passes, it will make the whole thing works
        print_sep()
        println("Dinv_omega0s_whole")
        display(reshape(Dinv_omega0s_whole[:,1], (2, nx, nt))[1,:,:])
        println()
        print_sep()
        println()

        print_sep()
        println("Dinv_omega0s_whole (approx)")
        display(reshape(BU[:,1], (2, nx, nt))[1,:, :])
        println()
        print_sep()
        println()

        COM1 = reshape(BU, (2, nx*nt, 2))
        COM2 = reshape(Dinv_omega0s_whole, (2, nx*nt, 2))

        diff = zero(Array{ComplexF64, 3}(undef, 2, lat_lambda0.ntot, 2))
        for i in 1:lat_lambda0.ntot
            diff[:, i, :] = (COM1[:, i, :] - COM2[:, i, :])./COM1[:, i, :]
        end
        println("DIFF")
        display(reshape([abs(i) for i in diff[1, :, 1]], (nx, lat_lambda0.nt)))
        println()
        asdF()
        """
        

        # Two different dirac components for source
        temp1 = Qwhole * Dinv_omega0s_whole[:, 1]
        temp2 = Qwhole * Dinv_omega0s_whole[:, 2]
        temp1 = reshape(temp1, (2, latcopy.ntot))
        temp2 = reshape(temp2, (2, latcopy.ntot))

        A = zero(Array{ComplexF64}(undef, 2, lattice.ntot, 2))
        
        """
        # The order of index ranges are VERY important
        # This has to be consistent with how with do sum in the latter update
        # and the order present below is correct (from small to large timeslices)
        A[:, :, 1] = cat(temp1[:, lat_lambda0.ntot+1:lat_lambda11.ntot+lat_lambda0.ntot], 
                         temp1[:, latcopy.ntot - lat_lambda12.ntot + 1:latcopy.ntot], 
                         dims=2)
        A[:, :, 2] = cat(temp2[:, lat_lambda0.ntot+1:lat_lambda11.ntot+lat_lambda0.ntot], 
                         temp2[:, latcopy.ntot - lat_lambda12.ntot + 1:latcopy.ntot],
                         dims=2)
        """

        # Cast to whole lattice
        A[:, :, 1] = temp1
        A[:, :, 2] = temp2
        
        

        if false
            # Only nonzero components are on the interior boundaries of lambda1
            # that are next to lambda0
            display(A[:, lat_lambda12.ntot - nx + 1: lat_lambda12.ntot, :])
            println()
            display(A[:, lat_lambda12.ntot+1:lat_lambda12.ntot+nx, :])
            println()
            # TODO: check if the rest elements are zeros
        end

        # Recall phi = zero(Array{ComplexF64, 4}(undef, n1, 2 , latcopy.ntot, 2))
        phi[in1_omega0s, :, :, :] = A

    end # n1 updates for omega0s

    # Now taking average over n1 configs
    phi = Statistics.mean(phi, dims=1)
    phin = phi[1, :, :, :] # mean is weird, it does not reduce the dimension automatically
    # Storing correlator before averaging
    n1_corr = zero(Array{ComplexF64, 2}(undef, n1, lat_lambda2.nt))

    # Loop over n1 updates of omega0s region
    print_sep()
    println("--> n1 updates of lambda2: total updates = $n1")
    print_sep()
    for in1_omega1s in 1:n1
        for ii in 1:n1_wait
            accprate = metropolis_update!(epsilon, lat_lambda2)
            println((Printf.@sprintf "Measurement iterations: %4d/%4d completed" in1_omega1s n1)*
                    (Printf.@sprintf ", current accp rate = %.2f)" accprate))
        end

        # Decompose domain again after each update
        decompose_domain()

        # Form sequential source
        s1 = phin[:, lat_lambda0.ntot+1:latcopy.ntot,:] # truncate from whole lattice
        # To ensure correctness, make sure the interior of lambda11 and lambda12 are all zeros
        for i in lat_lambda11.ntot
            if (i in 1:nx) == false || (i in lat_lambda11.ntot-nx+1:lat_lambda11.ntot) == false
                if s1[:, i, :] != zero(Array{ComplexF64, 2}(undef, 2, 2))
                    display(s1[:, i, :])
                    println()
                    error("s1 is not zero on lambda11 interiors (excluding boundaries)!")
                end
            end
        end
        for i in lat_lambda2.ntot+lat_lambda11.ntot+1:lat_omega1s.ntot
            if (i in lat_omega1s.ntot-nx+1:lat_omega1s.ntot) == false
                if s1[:, i, :] != zero(Array{ComplexF64, 2}(undef, 2, 2))
                    display(s1[:, i, :])
                    println()
                    error("s1 is not zero on lambda12 interiors (excluding boundaries)!")
                end
            end
        end
        # Now sequential inversion 
        Qomg1s = gamma5_Dslash_linearmap(lat_omega1s, lat_omega1s.mass)
        # Need to reshape to FlatField for solver
        s1inp1 = reshape(s1[:,:,1], (2*lat_omega1s.ntot))
        s1inp2 = reshape(s1[:,:,2], (2*lat_omega1s.ntot))
        seqprop = [minres_Q(Qomg1s, lat_omega1s, 
                            lat_omega1s.mass, s1inp1),
                   minres_Q(Qomg1s, lat_omega1s, 
                            lat_omega1s.mass, s1inp2)]

        # Trim sink to lambda2 only
        temp1 = reshape(seqprop[1], (2,lat_omega1s.ntot))[:, lat_lambda11.ntot+1:lat_lambda11.ntot+lat_lambda2.ntot]
        temp2 = reshape(seqprop[2], (2,lat_omega1s.ntot))[:, lat_lambda11.ntot+1:lat_lambda11.ntot+lat_lambda2.ntot]

        # Finally, we just need to tie those two ups
        tempret = zero(Array{ComplexF64, 2}(undef, lat_lambda2.nt, lat_lambda2.nt))
        fp = zero(Array{ComplexF64, 4}(undef, 2, nx, lat_lambda2.nt, 2))
        fp[:, :, :, 1] = reshape(temp1, (2, nx, lat_lambda2.nt))
        fp[:, :, :, 2] = reshape(temp2, (2, nx, lat_lambda2.nt))

        TensorOperations.@tensor begin
           tempret[t1, t2] = Gamma[a,c] * gamma5[c,d1] * fp[d1, ix, t1, b] * 
                             conj(fp[a, ix, t2, b2]) * Gamma[b,w] * gamma5[w,b2]
        end

        tempret = [tempret[ti,ti]/nx for ti in 1:lat_lambda2.nt]

        # Recall n1_corr = zero(Array{ComplexF64, 2}(undef, n1, lat_lambda2.nt))
        n1_corr[in1_omega1s, :] = tempret

    end # n1 updates for lambda2 region

    # Now we need to do n1 averaging
    n1_corr = Statistics.mean(n1_corr, dims=1) # n1 averaging
    n1_corr = n1_corr[1, :] 
    println()
    println("--> Current measurement: ")
    println(n1_corr)
    raw_corr[imol, :] = n1_corr
    println()
    # Average everything
    tt = Statistics.mean(raw_corr[1:imol, :], dims=1)
    println("--> Running average: ")
    println(tt[1, :])
    println()
end # n0 loop

# Average everything
tt = Statistics.mean(raw_corr, dims=1)
println("Final average: ")
println(tt[1, :])

# Save to file
NPZ.npzwrite(savname, raw_corr)
println("Saved to $(savname)")