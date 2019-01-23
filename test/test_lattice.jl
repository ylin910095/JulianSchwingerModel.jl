include("../src/lattice.jl")
include("../src/io.jl")

# Set up lattice parameters
nx = 5
nt = 4
kappa = 0.26                # hopping parameter
mass = (kappa^-1 - 4)/2
beta = 2.5
quenched = true

# Test out periodic boundary condition
bc1 = "periodic"

# Initilize lattice
lattice = Lattice(nx, nt, mass, beta, quenched, 
                  boundary_cond=bc1)

print_lattice(lattice)
println("Linear index (periodic):")
display(lattice.lin_indx)
println("\n")
println("Coordinate index (periodic):")
display(lattice.corr_indx)
println("\n")
println("upt array (periodic):")
display(lattice.upt)
println("\n")
println("downt array (periodic):")
display(lattice.downt)
println("\n")

# Test Dirichlet boundary condition using ghost cell
nt = 4
nx = 2

# Some fixed bc
# bc2 is a list of [anglex, anglet] for each ghost site.
# The first two in the list correspond to downt direction
# and the last two correspond to upt. The length of list has 
# to be 2*nx 
bc2 = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]

# Initilize lattice
lattice = Lattice(nx, nt, mass, beta, quenched, 
                  boundary_cond=bc2)

print_lattice(lattice)
println("Linear index (Dirichlet):")
display(lattice.lin_indx)
println("\n")
println("Coordinate index (Dirichlet):")
display(lattice.corr_indx)
println("\n")
println("upt array (Dirichlet):")
display(lattice.upt)
println("\n")
println("downt array (Dirichlet):")
display(lattice.downt)
println("\n")
println("anglex array (Dirichlet):")
display(lattice.anglex)
println("\n")
println("anglet array (Dirichlet):")
display(lattice.anglet)
println("\n")

# Now test trunc_lattice
lattice = Lattice(nx, nt, mass, beta, quenched, 
                  boundary_cond="periodic")
sublattice = trunc_lattice(lattice, 1:2)
print_lattice(sublattice)
println("Linear index (sublattice):")
display(sublattice.lin_indx)
println("\n")
println("Coordinate index (sublattice):")
display(sublattice.corr_indx)
println("\n")
println("upt array (sublattice):")
display(sublattice.upt)
println("\n")
println("downt array (sublattice):")
display(sublattice.downt)
println("\n")
println("anglex array (sublattice):")
display(sublattice.anglex)
println("\n")
println("anglet array (sublattice):")
display(sublattice.anglet)
println("\n")