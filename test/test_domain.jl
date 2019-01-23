using JulianSchwingerModel
using LinearAlgebra, Printf

"""
# Lattice info
DEBUG = true
nx = 32
nt = 32
kappa = 0.235                # hopping parameter
#kappa = 0.15
mass = (kappa^-1 - 4)/2
beta = 10.0
quenched = true

# Domains
lambda0 = 1:4
lambda2 = 18:23
lambda11 = 5:17
lambda12 = 24:32
"""

"""
# Lattice info
DEBUG = true
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

# Lattice info
DEBUG = true
nx = 32
nt = 32
kappa = 0.2               # hopping parameter
#kappa = 0.15
mass = (kappa^-1 - 4)/2
beta = 10.0
quenched = true

lambda0 = 1:12
lambda2 = 17:28
lambda11 = 13:16
lambda12 = 29:32


# Metropolis control
nthermal = 10
epsilon = 0.3

lattice = Lattice(nx, nt, mass, beta, quenched, 
                  boundary_cond="periodic")

print_lattice(lattice)
print_sep()
println("--> Begin Thermalization: total updates = $nthermal")
print_sep()
for i in 1:nthermal
    accprate = metropolis_update!(epsilon, lattice)
    println((Printf.@sprintf "Measurement iterations: %4d/%4d completed" i nthermal)*
            (Printf.@sprintf ", current accp rate = %.2f)" accprate))
end
global latcopy = deepcopy(lattice)

# Test the correctness of domian composition
# First truncat to sublattices
lat_lambda0 = trunc_lattice(lattice, lambda0)
lat_lambda2 = trunc_lattice(lattice, lambda2)
lat_lambda11 = trunc_lattice(lattice, lambda11)
lat_lambda12 = trunc_lattice(lattice, lambda12)
# Merge them back to see if they are still identitcal
# Reconstruct whole lattice
p = false # not periodic
safety_check = true
temp1 = stack_sublattice(lat_lambda0, lat_lambda11, p, safety_check)
temp2 = stack_sublattice(temp1, lat_lambda2, p, safety_check)
# Make sure periodic flag is true
lat_reconstruct = stack_sublattice(temp2, lat_lambda12, true, safety_check)

# Check the reconstruct lattice is identical to lattice
for i in lattice.ntot
    check1 = lattice.anglex[i] == lat_reconstruct.anglex[i]
    check2 = lattice.anglet[i] == lat_reconstruct.anglet[i]
    check3 = lattice.linkx[i] == lat_reconstruct.linkx[i]
    check4 = lattice.linkt[i] == lat_reconstruct.linkt[i]
    check5 = lattice.fermibc[i] == lat_reconstruct.fermibc[i]
    check6 = lattice.corr_indx[i] == lat_reconstruct.corr_indx[i]
    check7 = lattice.leftx[i] == lat_reconstruct.leftx[i]
    check8 = lattice.rightx[i] == lat_reconstruct.rightx[i]
    check9 = lattice.upt[i] == lat_reconstruct.upt[i]
    check10 = lattice.downt[i] == lat_reconstruct.downt[i]
    check11 = lattice.boundary_cond == lat_reconstruct.boundary_cond
    check_list = [check1, check2, check3, check4, check5, check6, check7, check8,
                  check9, check10]
    for ick in check_list
        if ick == false
            error(println("--> ERROR: check$i failed!"))
        end
    end
end
println("--> Domain decomposition passed!")



# Test propagators
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

# Forcing reconfigure all sublattices
decompose_domain(true)

# Domain stuff
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

# Make wall source at x0 in omega0s
x0 = 1
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

print_sep()
println("wallsource1, x0 = $(x0)")
display(reshape(wallsource1, 2, lat_omega0s.nx, lat_omega0s.nt)[1,:,:])
println()
print_sep()
println()

"""
for i in 1:10
    # See if domain update works
    # Well, aparently, it does not work
    metropolis_update!(epsilon, lat_omega0s)
end
"""

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
print_sep()
println("Dinv_omega0s")
display(reshape(Dinv_omega0s[1,:,1], (lat_omega0s.nx, lat_omega0s.nt)))
println()
print_sep()
println()


# A = \sum_{z}Q_{\Lambda_{10}}(w, z) * Q^{-1}_{\Omega_0^*}(z, x)
A = zero(Array{ComplexF64}(undef, 2, lat_omega0s.ntot - lat_lambda0.ntot, 2))
Dinv_omega0s_whole = zero(Array{ComplexF64}(undef, 2, latcopy.ntot, 2))
ic = 0

# Really careful of how you fill in the blank here
# since in omega0s, lambda12 comes first whereas it comes LAST
# in the whole lattice
# don't know how to line break...
Dinv_omega0s_whole[:,1:lat_lambda0.ntot,:] = Dinv_omega0s[:,lat_lambda12.ntot+1:lat_lambda12.ntot+lat_lambda0.ntot,:]


# Flatten it so we can use linearmap
Dinv_omega0s_whole = reshape(Dinv_omega0s_whole, (2*latcopy.ntot, 2))
BU = Dinv_omega0s_whole
Qwhole = gamma5_Dslash_linearmap(latcopy, latcopy.mass)


# Use exact form of propagator
# This will override Dinv_omega0s_whole computed before
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

    print_sep()
    println("wallsource1, x0 = $(x0)")
    display(reshape(wallsource1, 2, nx, nt)[1,:,:])
    println()
    print_sep()
    println()

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
end

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


# Two different dirac components for source
temp1 = Qwhole * Dinv_omega0s_whole[:, 1]
temp2 = Qwhole * Dinv_omega0s_whole[:, 2]
temp1 = reshape(temp1, (2, latcopy.ntot))
temp2 = reshape(temp2, (2, latcopy.ntot)) 

# The order of index ranges are VERY important
# This has to be consistent with how with do sum in the latter update
# and the order present below is correct (from small to large timeslices)
A[:, :, 1] = cat(temp1[:, lat_lambda0.ntot+1:lat_lambda11.ntot+lat_lambda0.ntot], 
                    temp1[:, latcopy.ntot - lat_lambda12.ntot + 1:latcopy.ntot], 
                    dims=2)
A[:, :, 2] = cat(temp2[:, lat_lambda0.ntot+1:lat_lambda11.ntot+lat_lambda0.ntot], 
                    temp2[:, latcopy.ntot - lat_lambda12.ntot + 1:latcopy.ntot],
                    dims=2)
print_sep()
println("A")
display(reshape(A[1,:,1], (lat_lambda12.nx, lat_lambda12.nt + lat_lambda11.nt)))
println()
print_sep()
println()
# Construct source
ic = 0
s1 = zero(Array{ComplexF64, 3}(undef, 2, lat_omega1s.ntot, 2)) # first source
for i in 1:lat_omega1s.ntot
    # Interior boundaries of lambda1 from omega1s
    if (i in 1:lat_lambda11.ntot) || (i in lat_lambda11.ntot
        +lat_lambda2.ntot+1:lat_omega1s.ntot)
        global ic += 1 
        s1[:, i, :] =  A[:, ic, :] 
    end
end 

# Sequential inversion
s1 = reshape(s1, (2*size(s1)[2], :))
print_sep()
println("s1")
display(reshape(s1[:, 1] ,(2, lat_omega1s.nx, lat_omega1s.nt))[1, :, :])
println()
print_sep()
println()
Q_omega1s = gamma5_Dslash_linearmap(lat_omega1s, lat_omega1s.mass)
seqprop = [minres_Q(Q_omega1s, lat_omega1s, lat_omega1s.mass, s1[:, 1]),
           minres_Q(Q_omega1s, lat_omega1s, lat_omega1s.mass, s1[:, 2])]

# truncate to lambda2
Dinv_lambda2_approx_1 = -reshape(seqprop[1], (2, lat_omega1s.ntot))[:, 
                        lat_lambda11.ntot+1:lat_lambda11.ntot+lat_lambda2.ntot]
Dinv_lambda2_approx_2 = -reshape(seqprop[2], (2, lat_omega1s.ntot))[:, 
                        lat_lambda11.ntot+1:lat_lambda11.ntot+lat_lambda2.ntot]

print_sep()
println("seqprop")
display(reshape(seqprop[1], (2, lat_omega1s.nx, lat_omega1s.nt))[1, :, :])
println()
print_sep()
println()

# Now do the inverse directly
# First construct wall source
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("Exact solution")

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

print_sep()
println("wallsource1, x0 = $(x0)")
display(reshape(wallsource1, 2, nx, nt)[1,:,:])
println()
print_sep()
println()

# Use the wallsource to invert propagator in omega0s domain
Q = gamma5_Dslash_linearmap(lattice, lattice.mass)
prop = [minres_Q(Q, lattice, 
                 lattice.mass, wallsource1),
        minres_Q(Q, lattice, 
                 lattice.mass, wallsource2)]

# Create matrix field as usual 
Dinv = zero(Array{ComplexF64}(undef, 2, lattice.ntot, 2))
tensorprop1 = reshape(prop[1], (2, lattice.ntot))
tensorprop2 = reshape(prop[2], (2, lattice.ntot))
Dinv[:, :, 1] = tensorprop1
Dinv[:, :, 2] = tensorprop2

# Truncate to lambda2
Dinv_lambda2 = Dinv[:, lat_lambda0.ntot+lat_lambda11.ntot+1:(nx*nt) - lat_lambda12.ntot,
                :]

print_sep()
println("Dinv")
display(reshape(Dinv[1,:,1], (nx, nt)))
println()
print_sep()
println()

print_sep()
println("Dinv_lambda2 (exact)")
display(reshape(Dinv_lambda2[1,:,1], (lat_lambda2.nx, lat_lambda2.nt)))
println()
print_sep()
println()


print_sep()
println("Dinv_lambda2 (approx)")
display(reshape(Dinv_lambda2_approx_1[1,:], (lat_lambda2.nx, lat_lambda2.nt)))
println()
print_sep()
println()

m1 = real(Dinv_lambda2[2,:,1])
m2 = real(Dinv_lambda2_approx_1[2,:])
diff = (m1 - m2)./m1
print_sep()
println("Percent difference ")
display(reshape([abs(i) for i in diff], (lat_lambda2.nx, lat_lambda2.nt)))
println()
print_sep()
println()
