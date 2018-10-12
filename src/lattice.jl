"""
Linear index (Coulumn-major) to tuple of coordinate in (x,t)
"""
@inline function lin2corr(i::Int64, nx::Int64)
    return (Int64((i-1)%nx + 1),
            Int64(floor((i-1)/nx) + 1))
end

"""
Copy all lattice content from lattice_from to lattice_to
"""
function deepcopy!(lattice_to::Lattice, lattice_from::Lattice)
    lattice_to.ntot = lattice_from.ntot
    lattice_to.nx = lattice_from.nx
    lattice_to.nt = lattice_from.nt
    lattice_to.mass = lattice_from.mass
    lattice_to.beta = lattice_from.beta
    lattice_to.quenched = lattice_from.quenched
    for i in 1:lattice_to.ntot
        lattice_to.lin_indx = lattice_from.lin_indx
        lattice_to.leftx = lattice_from.leftx
        lattice_to.rightx = lattice_from.rightx
        lattice_to.upt = lattice_from.upt
        lattice_to.downt = lattice_from.downt
        lattice_to.corr_indx = lattice_from.corr_indx
        lattice_to.anglex = lattice_from.anglex
        lattice_to.anglet = lattice_from.anglet
        lattice_to.linkx = lattice_from.linkx
        lattice_to.linkt = lattice_from.linkt
    end
end


"""
Called after updating gauge angles to make gauge links consistent
"""
function sync!(lattice::Lattice)
    for i in lattice.ntot
        lattice.linkx[i] = exp(lattice.anglex[i]*im)
        lattice.linkt[i] = exp(lattice.anglet[i]*im)
    end
end

function test_latticesetup(nx::Int64, nt::Int64, mass::Float64, beta::Float64, quenched::Bool)

    # Testing for correctness of finding neighbors
    @time lattice = Lattice(nx, nt, mass, beta, quenched)
    println("=======================================================================")
    println("=====                        Lattice Setup                        =====")
    println("=======================================================================")
    println("lattice dimensions (nx, nt): ", lattice.nx, ", ", lattice.nt)
    println("Coordinate indices: ")
    display(lattice.corr_indx)
    println("\n\nLinear indices")
    display(lattice.lin_indx)
    println("\n\nleft_x:")
    display(lattice.leftx)
    println("\n\nright_x: ")
    display(lattice.rightx)
    println("\n\nup_t:    ")
    display(lattice.upt)
    println("\n\ndown_t:  ")
    display(lattice.downt)
    println("")
    println("=======================================================================")
    println("=====                      Gauge Angles/Links                     =====")
    println("=======================================================================")
    println("Gauge angles in x direction: ")
    display(lattice.anglex)
    println("")
    println("Gauge angles in t direction: ")
    display(lattice.anglet)
    println("")
    println("Gauge links in x direction: ")
    display(lattice.linkx)
    println("")
    println("Gauge Angles in t direction: ")
    display(lattice.linkt)
    println("")
end
#test_latticesetup(5, 5, 0.1, 0.1)
