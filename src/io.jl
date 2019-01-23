function print_lattice(lattice::Lattice)
    println("Lattice Info:")
    print_sep()
    println("nx:         $(lattice.nx)")
    println("nt:         $(lattice.nt)")
    println("mass:       $(lattice.mass)")
    println("beta:       $(lattice.beta)")
    println("boundary:   $(lattice.boundary_cond)")
    println("quenched:   $(lattice.quenched)")
    print_sep()
end

"""
Print a line of dashes
"""
function print_sep()
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
    chsum = bytes2hex(SHA.sha256(checksumstring))
    return chsum
end

"""
Don't want to depend on other packages that might break in the future
Julia update. Save all information in lattice to filename in binary
"""
function save_lattice(lattice::Lattice, filename::String)
    io = open(filename, "w")

    # Calculate checksum for the lattice
    chsum = checksum_lattice(lattice::Lattice)

    # List of metainformation to be saved
    metalattice = [Int64(lattice.nx), Int64(lattice.nt),
                   Float64(lattice.mass),
                   Float64(lattice.beta), Int64(lattice.quenched),
                   Int64(length(chsum)), String(chsum)]
    for (ic, il) in enumerate(metalattice)
        write(io, il)
    end

    # Now store the gauge angles
    for i in 1:lattice.ntot
        write(io, Float64(lattice.anglet[i]))
    end

    for i in 1:lattice.ntot
        write(io, Float64(lattice.anglex[i]))
    end
    close(io)
end

function load_lattice(filename::String)
    buf = open(filename, "r")

    # Read line by line
    #allline = readlines(io, keep=true)
    #buf = IOBuffer(allline[1]) # Put into buffer so it can be parsed
    #println(length(allline))
    nx = read(buf, Int64)
    nt = read(buf, Int64)
    ntot = Int(nx * nt)
    mass = read(buf, Float64)
    beta = read(buf, Float64)
    quenched = Bool(read(buf, Int64))
    lenchsum = read(buf, Int64)
    checksum = ""
    for i in 1:lenchsum
        checksum *= read(buf, Char)
    end
    anglex = Array{Float64}(undef, ntot)
    anglet = Array{Float64}(undef, ntot)
    for i in 1:ntot
        anglet[i] = read(buf, Float64)
    end

    for i in 1:ntot
        anglex[i] = read(buf, Float64)
    end
    close(buf)

    # Finally, check integrity after loading
    lattice = Lattice(nx, nt, mass, beta, quenched, 
                      anglex0=anglex, anglet0=anglet)
    achsum = checksum_lattice(lattice)
    if achsum != checksum
        println(achsum)
        println(checksum)
        error("Inconsistent checksum after loading")
    end
    return lattice
end