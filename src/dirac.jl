using LinearAlgebra
using LinearMaps
using Base.Iterators

include("./lattice.jl")
include("./spinor.jl")
include("./gamma_matrices.jl")
include("./randlattice.jl")
include("./io.jl")

"""
gamma5*Dslash operator
Gauge fields have periodic boundary condition in all directions, whereas
fermions are periodic in spacelike direction (x-direction or 1 direction) and
antiperiodic in timelike direction (t-direction or 2 direction).
This is the fastest version of Dslash operates directly on the flatten spinor
array. Spinor of Field type can be obtained by
    field = collect(flatten(vector))
which requires using Base.Iterators
"""
function gamma5_Dslash_wilson_vector!(field_out::FlatField,
                                      field_in::FlatField,
                                      lattice::Lattice,
                                      mass::Float64)
    factor = 2 + mass
    for i in 1:lattice.ntot
        left1_i = lattice.leftx[i]
        left2_i = lattice.downt[i]
        right1_i = lattice.rightx[i]
        right2_i = lattice.upt[i]

        # Implementing periodic bc in space
        in_right1_1_i = field_in[dirac_comp1(right1_i)]
        in_right1_2_i = field_in[dirac_comp2(right1_i)]
        in_left1_1_i = field_in[dirac_comp1(left1_i)]
        in_left1_2_i = field_in[dirac_comp2(left1_i)]

        # Implementing antiperiodic bc in time
        if lattice.corr_indx[right2_i][2] == 1
            in_right2_1_i = -field_in[dirac_comp1(right2_i)]
            in_right2_2_i = -field_in[dirac_comp2(right2_i)]
        else
            in_right2_1_i = field_in[dirac_comp1(right2_i)]
            in_right2_2_i = field_in[dirac_comp2(right2_i)]
        end
        if lattice.corr_indx[left2_i][2] == lattice.nt
            in_left2_1_i = -field_in[dirac_comp1(left2_i)]
            in_left2_2_i = -field_in[dirac_comp2(left2_i)]
        else
            in_left2_1_i = field_in[dirac_comp1(left2_i)]
            in_left2_2_i = field_in[dirac_comp2(left2_i)]
        end

        link1 = lattice.linkx
        link2 = lattice.linkt
        cconj_link1_left1_i = conj(link1[left1_i])
        cconj_link2_left2_i = conj(link2[left2_i])

        field_out[dirac_comp1(i)] = factor * field_in[dirac_comp1(i)] - 0.5*(
                link1[i]*(in_right1_1_i - in_right1_2_i) +
                cconj_link1_left1_i * (in_left1_1_i + in_left1_2_i)  +
                link2[i] * (in_right2_1_i + im * in_right2_2_i) +
                cconj_link2_left2_i * (in_left2_1_i - im * in_left2_2_i)
                )
        field_out[dirac_comp2(i)] =  -factor * field_in[dirac_comp2(i)] - 0.5*(
            link1[i] * (in_right1_1_i - in_right1_2_i) -
            cconj_link1_left1_i * (in_left1_1_i  + in_left1_2_i)  +
            link2[i] * (im * in_right2_1_i - in_right2_2_i) -
            cconj_link2_left2_i * (im * in_left2_1_i  + in_left2_2_i)
            )
    end
end

function gamma5_Dslash_wilson_vector(field_in::FlatField,
                                     lattice::Lattice,
                                     mass::Float64)
    y = zero(FlatField(undef, 2*lattice.ntot))
    gamma5_Dslash_wilson_vector!(y, field_in, lattice, mass)
    return y
end

function gamma5_Dslash_wilson!(field_out::Field,
                               field_in::Field,
                               lattice::Lattice,
                               mass::Float64)
    tmpa = zero(FlatField(undef, 2 * lattice.ntot))
    gamma5_Dslash_wilson_vector!(tmpa, collect(flatten(field_in)),
                                 lattice, mass)
                                 
    for i in 1:lattice.ntot
        field_out[1, i] = tmpa[dirac_comp1(i)]
        field_out[2, i] = tmpa[dirac_comp2(i)]
    end
end

function gamma5_Dslash_wilson(field_in::Field,
                              lattice::Lattice,
                              mass::Float64)
    y = zero(Field(undef, 2, lattice.ntot))
    gamma5_Dslash_wilson!(y, field_in, lattice, mass)
    return y
end

"""
Another implementation of gamma5_Dslash but using
matrix multiplication instead of explicit assignment
to real and imaginary part.
This is slow and should not be used in production, for
validation of gamma5_Dslash_wilson only.
"""
function gamma5_Dslash_wilson_matrix(field_in::Field,
                                     lattice::Lattice,
                                     mass::Float64)
    field_out = zero(Field(undef, 2, lattice.ntot))
    factor = 2 + mass
    for i in 1:lattice.ntot
        left1_i = lattice.leftx[i]
        left2_i = lattice.downt[i]
        right1_i = lattice.rightx[i]
        right2_i = lattice.upt[i]
        link1 = lattice.linkx
        link2 = lattice.linkt
        # Implementing periodic bc in space
        in_right1_i = field_in[:, right1_i]
        in_left1_i = field_in[:, left1_i]
        # Implementing antiperiodic bc in time
        if lattice.corr_indx[right2_i][2] == 1
            in_right2_i = -field_in[:, right2_i]
        else
            in_right2_i = field_in[:, right2_i]
        end
        if lattice.corr_indx[left2_i][2] == lattice.nt
            in_left2_i = -field_in[:, left2_i]
        else
            in_left2_i = field_in[:, left2_i]
        end
        
        field_out[:, i] = gamma5 * (factor .* field_in[:, i] - 0.5*(
            link1[i] .* (I - gamma1) * in_right1_i +
            conj(link1[left1_i]) .* (I + gamma1) * in_left1_i +
            link2[i] .* (I - gamma2) * in_right2_i +
            conj(link2[left2_i]) .* (I + gamma2) * in_left2_i
            ))
        
    end
    return field_out
end

"""
Convert gamma5_Dslash to LinearMap type for a given gauge configuration.
This wrapper is required for input to IterativeSolvers and it is not optimal.
"""
function gamma5_Dslash_linearmap(lattice::Lattice, mass::Float64)
    Q(v::FlatField) = gamma5_Dslash_wilson_vector(v, lattice, mass)
    g5D = LinearMap{ComplexF64}(Q, nothing, 2*lattice.ntot,
                                2*lattice.ntot; ishermitian=true)
    return g5D
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
    flat_in = collect(flatten(field_in))
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

#test_gamma5Dslash()
