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
thermaliter = 1000
iterwait = 200 # number of updates between each saved configuration

# Naming convention for saved file
savdir = "../gauge_metropolis"
series = "b"
start_no = 1
tot_gauge = 1000 # number of total gauge configurations to generate
epsilon = 0.3
prefix = @sprintf "%s/l%d%db%.4fk%.4f" savdir nx nt beta kappa 

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
    for j in 1:iterwait
        accprate = metropolis_update!(epsilon, lattice)
        println((Printf.@sprintf "Meas iterations (metropolis): %4d/%4d completed (%4d/%4d wait)" i thermaliter j iterwait)*
                (Printf.@sprintf ", current accp rate = %.4f" accprate))
    end
end