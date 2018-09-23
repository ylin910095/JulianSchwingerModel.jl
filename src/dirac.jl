using LinearAlgebra
using LinearMaps

include("./lattice.jl")
include("./spinor.jl")
include("./gamma_matrices.jl")
include("./randlattice.jl")

"""
gamma5*Dslash operator
Gauge fields have periodic boundary condition in all directions, whereas
fermions are periodic in spacelike direction (x-direction or 1 direction) and 
antiperiodic in timelike direction (t-direction or 2 direction)
"""
function gamma5_Dslash_wilson(field_in::Field, 
                              lattice::Lattice,
                              mass::Float64)
    factor = 2 + mass 
    field_out = Field(undef, lattice.ntot)
    zero!(field_out)
    for i in 1:lattice.ntot
        left1_i = lattice.leftx[i]
        left2_i = lattice.downt[i]
        right1_i = lattice.rightx[i]
        right2_i = lattice.upt[i]

        # Implementing periodic bc in space
        in_right1_i = field_in[right1_i]
        in_left1_i = field_in[left1_i]
        
        # Implementing antiperiodic bc in time
        if lattice.corr_indx[right2_i][2] == 1
            in_right2_i = -field_in[right2_i]
        else
            in_right2_i = field_in[right2_i]
        end
        if lattice.corr_indx[left2_i][2] == lattice.nt
            in_left2_i = -field_in[left2_i]
        else
            in_left2_i = field_in[left2_i]
        end
        
        link1 = lattice.linkx
        link2 = lattice.linkt
        cconj_link1_left1_i = conj(link1[left1_i])
        cconj_link2_left2_i = conj(link2[left2_i])
        
        # First spinor component
        field_out[i][1] = factor * field_in[i][1] - 0.5*(
                link1[i]*(in_right1_i[1] - in_right1_i[2]) +
                cconj_link1_left1_i * (in_left1_i[1] + in_left1_i[2])  +
                link2[i] * (in_right2_i[1] + im * in_right2_i[2]) +
                cconj_link2_left2_i * (in_left2_i[1] - im * in_left2_i[2])
                )
        # Second spinor component
        field_out[i][2] = -factor * field_in[i][2] - 0.5*(
            link1[i] * (in_right1_i[1] - in_right1_i[2]) -
            cconj_link1_left1_i * (in_left1_i[1]  + in_left1_i[2])  +
            link2[i] * (im * in_right2_i[1] - in_right2_i[2]) -
            cconj_link2_left2_i * (im * in_left2_i[1]  + in_left2_i[2])
            )
    end
    return field_out
end

"""
Another implementation of gamma5_Dslash but using 
matrix multiplication instead of explicit assignment
to real and imaginary part.
This is slow and should not be used in production.
"""
function gamma5_Dslash_wilson_matrix(field_in::Field, 
                                     lattice::Lattice,
                                     mass::Float64)
    field_out = Field(undef, lattice.ntot)
    zero!(field_out)
    factor = 2 + mass
    for i in 1:lattice.ntot
        left1_i = lattice.leftx[i]
        left2_i = lattice.downt[i]
        right1_i = lattice.rightx[i]
        right2_i = lattice.upt[i]
        link1 = lattice.linkx
        link2 = lattice.linkt
        # Implementing periodic bc in space
        in_right1_i = field_in[right1_i]
        in_left1_i = field_in[left1_i]
        # Implementing antiperiodic bc in time
        if lattice.corr_indx[right2_i][2] == 1
            in_right2_i = -field_in[right2_i]
        else
            in_right2_i = field_in[right2_i]
        end
        if lattice.corr_indx[left2_i][2] == lattice.nt
            in_left2_i = -field_in[left2_i]
        else
            in_left2_i = field_in[left2_i]
        end
        field_out[i] = gamma5 * (factor .* field_in[i] - 0.5*(
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
    # We need to unravel nested array to use IterativeSolvers
    A(v::Vector{ComplexF64}) = unravel(gamma5_Dslash_wilson(ravel(v), lattice, mass))
    g5D = LinearMap{ComplexF64}(A, nothing, 2*lattice.ntot, 2*lattice.ntot; ishermitian=true)
    return g5D
end

function test_gamma5Dslash()
    nx = 5
    nt = 20
    mass = 0.02
    beta = 0.1
    outputflag = 0 # 0 for no error, nonzero for more than one errors
    lattice = Lattice(nx, nt, mass, beta)
    spinor_in = Spinor(lattice.ntot)
    for i in 1:lattice.ntot
        spinor_in.s[i] = [gauss(), gauss()]
    end

    # Check correctness
    println("=======================================================================")
    println("=====                    gamma_5*Dslash Wilson                    =====")
    println("=======================================================================")
    println("Lattice dimension: (nx=$nx, nt=$nt), mass: $mass")
    print("gamma_5 * Dslash explicit form: ")
    @time field_out1 = gamma5_Dslash_wilson(spinor_in.s, lattice, mass)
    print("gamma_5 * Dslash matrix form: ")
    @time field_out2 = gamma5_Dslash_wilson_matrix(spinor_in.s, lattice, mass)
    ddiff = 0.0
    for i in 1:lattice.ntot
        ddiff += field_out1[i][1] - field_out2[i][1]
        ddiff += field_out1[i][2] - field_out2[i][2]
    end
    if ddiff == 0
        println("COMPARISON PASSED: gamma5_Dslash_wilson vs gamma5_Dslash_wilson_matrix")
    else
        println("COMPARISON FAILED: gamma5_Dslash_wilson vs gamma5_Dslash_wilson_matrix")
        outputflag += 1
    end

    # Now check hermiticity of gamma5*Dslash by constructing
    # the full matrix representation (only work for small ntot)
    Dslash = Array{ComplexF64, 4}(undef, lattice.ntot, lattice.ntot, 2, 2)
    spinor_in = Spinor(lattice.ntot)
    spinor_out = Spinor(lattice.ntot)

    for i in 1:lattice.ntot
        for j in 1:lattice.ntot
            for k in 1:2
                for l in 1:2
                    spinor_in.s[j][l] = 1.0
                    spinor_out.s = gamma5_Dslash_wilson(spinor_in.s, lattice, 0.02)
                    spinor_in.s[j][l] = 0.0 # reset to zero
                    Dslash[i,j,k,l] = spinor_out.s[i][k]
                end
            end
        end
    end
    
    hermiflag = 0
    for i in 1:lattice.ntot
        for j in 1:lattice.ntot
            for k in 1:2
                for l in 1:2
                    if Dslash[i,j,k,l] != conj(Dslash[j,i,l, k])
                        hermiflag += 1
                    end
                end
            end
        end
    end
    if hermiflag == 0
        println("HERMITICITY PASSED: gamma5_Dslash_wilson")
    else
        println("HERMITICITY FAILED: gamma5_Dslash_wilson")
    end
    outputflag += hermiflag

    
    # Test linearmap construction of gamma5_Dslash_wilson
    g5D = gamma5_Dslash_linearmap(lattice, mass)
    spinor_in = Spinor(lattice.ntot)
    for i in 1:lattice.ntot
        spinor_in.s[i] = [gauss(), gauss()]
    end

    print("gamma_5 * Dslash LinearMap form: ")
    @time field_out = ravel(g5D * unravel(spinor_in.s))
    print("gamma_5 * Dslash explicit form: ")
    @time spinor_out.s = gamma5_Dslash_wilson(spinor_in.s, lattice, mass)

    ddiff = 0
    for i in 1:lattice.ntot
        ddiff += field_out[i][1]-spinor_out.s[i][1]
        ddiff += field_out[i][2]-spinor_out.s[i][2]
    end
    if ddiff == 0
        println("CORRECTNESS PASSED: gamma5_Dslash_linearmap ")
    else
        println("CORRECTNESS FAILED: gamma5_Dslash_linearmap")
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