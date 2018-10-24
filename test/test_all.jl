"""
Temporary file to put all tests. Scripts don't work yet.
"""

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

function test_gamma5Dslash()
    nx = 10
    nt = 10
    mass = 0.02
    beta = 0.1
    quenched = false
    outputflag = 0 # 0 for no error, nonzero for more than one errors
    lattice = Lattice(nx, nt, mass, beta, quenched)
    field_in = Field(undef, 2, lattice.ntot)
    for i in 1:2*lattice.ntot
        field_in[i] = gauss() + im * gauss()
    end

    # Check correctness
    print_lattice(lattice)
    println()
    println("=======================================================================")
    println("=====                    gamma_5*Dslash Wilson                    =====")
    println("=======================================================================")
    print("gamma_5 * Dslash explicit form: ")
    @time field_out1 = gamma5_Dslash_wilson(field_in, lattice, mass)
    print("gamma_5 * Dslash matrix form: ")
    @time field_out2 = gamma5_Dslash_wilson_matrix(field_in, lattice, mass)
    ddiff = sum(field_out1 - field_out2)
    if ddiff == 0
        println("COMPARISON PASSED: gamma5_Dslash_wilson vs gamma5_Dslash_wilson_matrix")
    else
        println("COMPARISON FAILED: gamma5_Dslash_wilson vs gamma5_Dslash_wilson_matrix, difference = $ddiff")
        outputflag += 1
    end

    ################################################################
    # Now check hermiticity of gamma5*Dslash by constructing
    # the full matrix representation
    Q = zero(Array{ComplexF64}(undef, 2*lattice.ntot, 2*lattice.ntot))
    x = zero(FlatField(undef, 2*lattice.ntot))
    for i in 1:2*lattice.ntot
        x[i] = 1.0 
        Q[:, i] = gamma5_Dslash_wilson_vector(x, lattice, lattice.mass)
        x[i] = 0.0 
    end 

    # Check Hermiticity
    dQ = adjoint(Q) - Q
    if dQ == zero(Array{ComplexF64}(undef, 2*lattice.ntot, 2*lattice.ntot))
        println("HERMITICITY PASSED: gamma5_Dslash_wilson")
    else
        println("HERMITICITY FAILED: gamma5_Dslash_wilson")
        outputflag += 1
    end
    ################################################################
    # Test linearmap construction of gamma5_Dslash_wilson
    g5D = gamma5_Dslash_linearmap(lattice, mass)
    field_in = Field(undef, 2, lattice.ntot)
    for i in 1:2*lattice.ntot
        field_in[i] = gauss() + im * gauss()
    end
    print("gamma_5 * Dslash LinearMap form: ")
    flat_in = collect(Base.Iterators.flatten(field_in))
    @time temp_out = g5D * flat_in
    field_out1 = reshape(temp_out, (2, lattice.ntot))
    print("gamma_5 * Dslash explicit form: ")
    @time field_out2 = gamma5_Dslash_wilson(field_in, lattice, mass)

    ddiff = sum(field_out1 - field_out2)
    if ddiff == 0
        println("CORRECTNESS PASSED: gamma5_Dslash_linearmap ")
    else
        println("CORRECTNESS FAILED: gamma5_Dslash_linearmap, difference = $ddiff")
        outputflag += 1
    end

    if outputflag == 0
        println("ALL TESTS PASSED")
    else
        println("SOME TESTS FAILED")
    end
    return outputflag
end

function test_HMC()
    # Lattice param
    nx = 32
    nt = 32
    kappa = 0.26 # Hopping parameter
    mass = (kappa^-1 - 4)/2
    beta = 2.5
    quenched = true

    # HMC param
    tau = 3
    integrationsteps = 200
    hmciter = 10000
    thermalizationiter = 1000
    lattice = Lattice(nx, nt, mass, beta, quenched)
    hmcparam = HMCParam(tau, integrationsteps, hmciter, thermalizationiter)

    accptot = 0
    # Thermalization
    for ithiter in 1:hmcparam.thermalizationiter
        println("Thermalization steps: $ithiter/$(hmcparam.thermalizationiter)")
        accp = HMCWilson_update!(lattice, hmcparam)
        accptot += accp
        accprate = accptot/ithiter
        plaq = measure_wilsonloop(lattice)
        println("Accept = $accp; Acceptance rate = $accprate; Plaquette = $plaq")
    end
    # Actual measurements
    plaqsum = 0.0
    accptot_hmc = 0
    for ihmciter in 1:hmcparam.niter
        println("Measurement steps: $ihmciter/$(hmcparam.niter)")
        accp = HMCWilson_update!(lattice, hmcparam)
        accptot += accp
        accptot_hmc += accp
        accprate = accptot/(ihmciter + hmcparam.thermalizationiter)
        if accp == 1
            plaq = measure_wilsonloop(lattice)
            plaqsum += plaq
            println("Accept = $accp; Acceptance rate = $accprate; Plaquette = $plaq")
            println("Avg plaq: $(plaqsum/accptot_hmc)")
        end
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

function test_minres()
    nx = 10
    nt = 10
    mass = 0.1
    beta = 2.0
    quenched = false
    # TEST1: Does minres gives desired accuracy
    lattice = Lattice(nx, nt, mass, beta, quenched)
    source = zero(FlatField(undef, 2*lattice.ntot))
    # Random source
    for i in 1:length(source)
        source[i] = gauss() + im * gauss()
    end
    # Constructing linear map
    Q = gamma5_Dslash_linearmap(lattice, lattice.mass)
    field_out = minres_Q(Q, lattice, mass, source, false)

    # Test to see if the solutions have converged
    y = gamma5_Dslash_wilson_vector(field_out, lattice, mass)
    ddiff = sum(y - source)
    println("TEST1: Final difference = $ddiff")


    print_sep()
    # TEST2: What happens if we change lattice, does Q changes too?
    lattice = Lattice(nx, nt, mass, beta, quenched)
    source = zero(FlatField(undef, 2*lattice.ntot))
    # Random source
    for i in 1:length(source)
        source[i] = gauss() + im * gauss()
    end
    Q = gamma5_Dslash_linearmap(lattice, lattice.mass)
    # Now change lattice
    for i in 1:lattice.ntot
        lattice.anglex[i] = 2pi * rand01()
        lattice.anglet[i] = 2pi * rand01()
    end
    sync!(lattice)
    out1 = Q * source
    out2 = gamma5_Dslash_wilson_vector(source, lattice, lattice.mass)
    ddiff = sum(out1 - out2)
    println("TEST2: Final difference = $ddiff")
    print_sep()
end

function test_leapfrog()
    lattice = Lattice(3, 3, 0.1, 0.1)
    pf = PseudoFermion(lattice, gamma5_Dslash_wilson)
    p = HMCMom(lattice)
    leapfrog!(p, pf, 10, 1.0, lattice)
end