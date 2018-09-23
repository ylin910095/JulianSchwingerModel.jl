using SHA
include("./lattice.jl")
include("./spinor.jl")

function print_lattice(lattice::Lattice)
    println("Lattice Info:")
    println("-------------------------------")
    println("nx:         $(lattice.nx)")
    println("nt:         $(lattice.nt)")
    println("mass:       $(lattice.mass)")
    println("quenched:   $(lattice.quenched)")
    println("-------------------------------")
end

"""
Calculate the checksum of lattice using SHA.
"""
function checksum_lattice(lattice::Lattice)
    checksumstring = ""
    metalattice = [lattice.nx, lattice.nt, lattice.mass,
                   lattice.beta, lattice.quenched]
    # Metainformation
    for ien in metalattice
        checksumstring *= string(ien)
    end
    # Gauge angles
    for i in 1:lattice.ntot
        checksumstring *= string(lattice.anglex[i])
        checksumstring *= string(lattice.anglet[i])
    end
    chsum = bytes2hex(sha256(checksumstring))
    return chsum
end

"""
Don't want to depend on other packages that might break in the future
Julia update. Save all information in lattice to filename
"""
function save_lattice(lattice::Lattice, filename::String)
    io = open(filename, "w") 

    # Calculate checksum for the lattice 
    chsum = checksum_lattice(lattice::Lattice)

    # List of metainformation to be saved
    metalattice = [lattice.nx, lattice.nt, lattice.mass,
                   lattice.beta, Int64(lattice.quenched), chsum]
    for (ic, il) in enumerate(metalattice)
        write(io, il)
    end
    write(io, "\n")

    # Now store the gauge angles
    for i in 1:lattice.ntot
        write(io, lattice.anglet[i])
    end
    write(io, "\n")

    for i in 1:lattice.ntot
        write(io, lattice.anglex[i])
    end
    write(io, "\n")

    close(io)
end

function load_lattice(filename::String)
    io = open(filename, "r")

    # Read line by line
    allline = readlines(io)
    buf = IOBuffer(allline[1]) # Put into buffer so it can be parsed
    #println(length(allline))

    nx = read(buf, Int64)
    nt = read(buf, Int64)
    ntot = Int(nx * nt)
    mass = read(buf, Float64)
    beta = read(buf, Float64)
    quenched = Bool(read(buf, Int64))
    checksum = read(buf, String)
    anglex = Array{Float64}(undef, ntot)
    anglet = Array{Float64}(undef, ntot)

    # Read gauge angles
    buf = IOBuffer(allline[2])
    for i in 1:ntot
        anglex[i] = read(buf, Float64)
    end
    buf = IOBuffer(allline[3])
    for i in 1:ntot
        anglet[i] = read(buf, Float64)
    end
    
    close(io)
    return Lattice(nx, nt, mass, beta, quenched, anglex, anglet)
end

function test_latticeio()
    # Lattice param
    nx = 32
    nt = 32
    kappa = 0.26 # Hopping parameter
    mass = (kappa^-1 - 4)/2
    beta = 2.5
    quenched = true
    lattice = Lattice(nx, nt, mass, beta, quenched)
    chsum = checksum_lattice(lattice)
    save_lattice(lattice, "testsave.txt")
    returned_lattice = load_lattice("testsave.txt")

    println("Before saving:")
    print_lattice(lattice)
    println()
    println("After saving")
    print_lattice(returned_lattice)
    if checksum_lattice(lattice) == checksum_lattice(returned_lattice)
        println("INTEGRITY CHECKED: Checksums are consistent")
    else
        error("INTEGRITY FAILED: Checksums are different")
    end
end