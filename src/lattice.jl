# Strcuture of Arrays
mutable struct Lattice
    # Inputs
    nx::Int64
    nt::Int64
    mass::Float64 # Sea quarks masses
    beta::Float64
    quenched::Bool

    # Initilize derived variables
    ntot::Int64
    lin_indx::Array{Int64, 2}
    leftx::Array{Int64, 2}
    rightx::Array{Int64, 2}
    upt::Array{Int64, 2}
    downt::Array{Int64, 2}
    corr_indx::Array{Tuple{Int64, Int64}, 2} # coordinate tuple in the order of (x,t)

    # Gauge link and gauge angles
    anglex::Array{Float64}
    anglet::Array{Float64}
    linkx::Array{ComplexF64, 2}
    linkt::Array{ComplexF64, 2}

    # Inner constructor method to initilize structure
    function Lattice(nx::Int64, nt::Int64, mass::Float64, beta::Float64, quenched::Bool)
        ntot = nx * nt
        leftx  = Array{Int64, 2}(undef, nx, nt)
        rightx = Array{Int64, 2}(undef, nx, nt)
        upt = Array{Int64, 2}(undef, nx, nt)
        downt = Array{Int64, 2}(undef, nx, nt)
        corr_indx = Array{Tuple{Int64, Int64}, 2}(undef, nx, nt)
        lin_indx = Array{Int64, 2}(undef, nx, nt)

        # Gauge stuff
        anglex = Array{Float64, 2}(undef, nx, nt)
        anglet = Array{Float64, 2}(undef, nx, nt)
        linkx = Array{ComplexF64, 2}(undef, nx, nt)
        linkt = Array{ComplexF64, 2}(undef, nx, nt)


        for i in 1:ntot
            # Find the closest neighbors and coordinate indices
            lin_indx[i] = i
            if (i-1) % nx != 0
                leftx[i] = i - 1 
            else
                leftx[i] = i + nx - 1 
            end
            if i % nx != 0
                rightx[i] = i + 1
            else
                rightx[i] = i - nx + 1
            end
            upt[i] = i + nx
            if upt[i] > ntot
                upt[i] -= ntot
            end 
            downt[i] = i - nx
            if downt[i] < 1
                downt[i] += ntot
            end
            corr_indx[i] = ((i-1)%nx + 1, 
                            floor((i-1)/nx) + 1)

            # Cold start for gauge links
            anglex[i] = 0.0
            anglet[i] = 0.0
            linkx[i] = exp(anglex[i]*im)
            linkt[i] = exp(anglet[i]*im)

        end
        # The ordering here corresponds to the ordering of definition 
        # at the beginning of struct. If wrongly ordered, it will raise
        # errors
        new(nx, nt, mass, beta, quenched,
            ntot, lin_indx, leftx, rightx, upt, downt, corr_indx, 
            anglex, anglet, linkx, linkt)
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
