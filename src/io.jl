using SHA
include("./lattice.jl")
include("./spinor.jl")
include("./randlattice.jl")
include("./measurements.jl")

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
    """
    # Now combine all the rest of lines into a single string
    # This is because \n could appear in binary encoding,
    # and it will cause errors if we separate them line by line
    allbinstr = ""
    for istr in allline[2:end]
        println(istr)
        allbinstr *= istr
    end
    # Read gauge angles
    buf = IOBuffer(allbinstr)
    """
    for i in 1:ntot
        anglet[i] = read(buf, Float64)
    end
    
    for i in 1:ntot
        anglex[i] = read(buf, Float64)
    end
    close(buf)

    # Finally, check integrity after loading
    lattice = Lattice(nx, nt, mass, beta, quenched, anglex, anglet)
    achsum = checksum_lattice(lattice)
    if achsum != checksum
        println(achsum)
        println(checksum)
        error("Inconsistent checksum after loading")
    end
    return lattice
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
    # Randomize lattice
    for i in 1:lattice.ntot
        lattice.anglex[i] = gauss()
        lattice.anglet[i] = gauss()
    end
    save_lattice(lattice, "testsave.txt")
    returned_lattice = load_lattice("testsave.txt")

    println("Before saving:")
    print_lattice(lattice)
    println()
    println("After saving")
    print_lattice(returned_lattice)
    println("TESTS PASSED: IO ")
end
#test_latticeio()