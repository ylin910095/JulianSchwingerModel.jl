using JulianSchwingerModel
using NPZ, Printf

# Lattice info
nx = 32
nt = 32
kappa = 0.2               
mass = (kappa^-1 - 4)/2
beta = 10.0
quenched = true

# Gauge generation parameter
epsilon = 0.5 # metropolis update parameter
thermaliter = 1
iterwait = 200 # number of updates between each saved config

# Naming convention for saved file
savdir = "../gauge_metropolis"
series = "b"
start_no = 1
tot_gauge = 1000 # number of total gauge configurations to generate
epsilon = 0.3
prefix = @sprintf "%s/l%d%db%.4fk%.4f" savdir nx nt beta kappa 

# For domain decomposition
generate_domain = true
if generate_domain
        n1 = 20 # number of domains updates per top lelvel configuration
        n1_wait = 50 # number of updates between each n1 level saved config
        # Need to define domains to update. The codes will update ONLY
        # lambda0 and lambda2 while freezing lambda11 and lambda12
        lambda0 = 1:12
        lambda2 = 17:28
        lambda11 = 13:16
        lambda12 = 29:32
end

# Do work
@assert (typeof(start_no) == Int) && (start_no > 0) "start_no has to be positive integer! (current values: $start_no)" 
lattice = Lattice(nx, nt, mass, beta, quenched)
print_sep()
println("--> Begin Thermalization: total updates = $thermaliter")
print_sep()
for i in 1:thermaliter
    accprate = metropolis_update!(epsilon, lattice)
    println((Printf.@sprintf "Thermal iterations (metropolis): %4d/%4d completed" i thermaliter)*
            (Printf.@sprintf ", current accp rate = %.4f" accprate))
end


print_lattice(lattice)
print_sep()
println("--> Begin Measurements: total updates = $tot_gauge")
print_sep()
for i in 1:tot_gauge
    savname = "$(prefix)_$(series)$(start_no+i-1).metro"
    save_lattice(lattice, savname)
    println("--> Saved to: $savname")

    # Lower level updates if requested
    if generate_domain 
        lat_lambda0 = trunc_lattice(lattice, lambda0)
        lat_lambda2 = trunc_lattice(lattice, lambda2)
        lat_lambda11 = trunc_lattice(lattice, lambda11)
        lat_lambda12 = trunc_lattice(lattice, lambda12)
        println("--> Begin sublattice generation: total updates = $n1")
        for in1 in 1:n1
            # Don't saved the same lattice twice so we update sublattices
            # first in the for loop
            for ii in 1:n1_wait
                accprate1 = metropolis_update!(epsilon, lat_lambda0)
                accprate2 = metropolis_update!(epsilon, lat_lambda2)
                println((Printf.@sprintf "Meas iterations (metropolis): %4d/%4d completed (%4d/%4d wait)" in1 n1 ii n1_wait)*
                (Printf.@sprintf ", current accp rate (lambda0) = %.4f" accprate1)*(Printf.@sprintf ", current accp rate (lambda2) = %.4f" accprate2)) 
            end

            # Now restack the lattices and save it
            # Enable safety check ONLY if we first truncate lattice
            # When we later update partial domain, say lambda0,
            # then the ghost cells will not be updated in lambda11
            # and lambda12 and sanity check will fail when we try to 
            # stack them.
            safety_check = false
            periodic = false
            temp1 = stack_sublattice(lat_lambda0, lat_lambda11, periodic, safety_check)
            temp2 = stack_sublattice(temp1, lat_lambda2, periodic, safety_check)
            # Make sure periodic flag is true when reconstructing whole lattice
            periodic = true
            latcopy = stack_sublattice(temp2, lat_lambda12, periodic, safety_check)

            subsavname = "$(prefix)_$(series)$(start_no+i-1)_sub$(in1).metro"
            save_lattice(latcopy, subsavname)
            println("--> Sublattice saved to: $subsavname")
        end
    end # n1 updates

    # Top level updates
    for j in 1:iterwait
        accprate = metropolis_update!(epsilon, lattice)
        println((Printf.@sprintf "Meas iterations (metropolis): %4d/%4d completed (%4d/%4d wait)" i thermaliter j iterwait)*
                (Printf.@sprintf ", current accp rate = %.4f" accprate))

    end 
end