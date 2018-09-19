using LinearAlgebra

include("./lattice.jl")
include("./gamma_matrices.jl")

function gamma5_Dslash_wilson(spinor_in::Spinor, 
                              lattice::Lattice,
                              mass::Float64)
    factor = 2 + mass
    spinor_out = Spinor(lattice.ntot)
    for i in 1:lattice.ntot
        left1_i = lattice.leftx[i]
        left2_i = lattice.downt[i]
        right1_i = lattice.rightx[i]
        right2_i = lattice.upt[i]
        in_right1_i = spinor_in.s[right1_i]
        in_right2_i = spinor_in.s[right2_i]
        in_left1_i = spinor_in.s[left1_i]
        in_left2_i = spinor_in.s[left2_i]
        link1 = lattice.linkx
        link2 = lattice.linkt
        cconj_link1_left1_i = conj(link1[left1_i])
        cconj_link2_left2_i = conj(link2[left2_i])

        # First spinor component
        spinor_out.s[i][1] = factor * spinor_in.s[i][1] - 0.5*(
                link1[i]*(in_right1_i[1] - in_right1_i[2]) +
                cconj_link1_left1_i * (in_left1_i[1] + in_left1_i[2])  +
                link2[i] * (in_right2_i[1] + im * in_right2_i[2]) +
                cconj_link2_left2_i * (in_left2_i[1] - im * in_left2_i[2])
                )
        # Second spinor component
        spinor_out.s[i][2] = -factor * spinor_in.s[i][2] - 0.5*(
            link1[i] * (in_right1_i[1] - in_right1_i[2]) -
            cconj_link1_left1_i * (in_left1_i[1]  + in_left1_i[2])  +
            link2[i] * (im * in_right2_i[1] - in_right2_i[2]) -
            cconj_link2_left2_i * (im * in_left2_i[1]  + in_left2_i[2])
            )
    end
    return spinor_out
end

"""
Another implementation of gamma5_Dslash but using 
matrix multiplication instead of explicit assignment
to real and imaginary part.
This is slow and should not be used in production.
"""
function gamma5_Dslash_wilson_matrix(spinor_in::Spinor, 
                                     lattice::Lattice,
                                     mass::Float64)
    factor = 2 + mass
    spinor_out = Spinor(lattice.ntot)
    for i in 1:lattice.ntot
        left1_i = lattice.leftx[i]
        left2_i = lattice.downt[i]
        right1_i = lattice.rightx[i]
        right2_i = lattice.upt[i]
        link1 = lattice.linkx
        link2 = lattice.linkt
        spinor_out.s[i] = gamma5 * (factor .* spinor_in.s[i] - 0.5*(
            link1[i] .* (I - gamma1) * spinor_in.s[right1_i] +
            conj(link1[left1_i]) .* (I + gamma1) * spinor_in.s[left1_i] +
            link2[i] .* (I - gamma2) * spinor_in.s[right2_i] + 
            conj(link2[left2_i]) .* (I + gamma2) * spinor_in.s[left2_i]
            ))
    end
    return spinor_out
end
function test_gamma5Dslash()
    nx = 50
    nt = 50
    mass = 0.02
    lattice = Lattice(nx, nt, mass)
    spinor_in = Spinor(lattice.ntot)
    for i in 1:lattice.ntot
        spinor_in.s[i][1] = randn(Float64)
        spinor_in.s[i][2] = randn(Float64)
    end
    println("=======================================================================")
    println("=====                      gamma_5*Dslash Wilson                  =====")
    println("=======================================================================")
    println("Lattice dimension: (nx=$nx, nt=$nt), mass: $mass")
    print("gamma_5 * Dslash explicit form: ")
    @time spinor_out_1 = gamma5_Dslash_wilson(spinor_in, lattice, 0.02)
    print("gamma_5 * Dslash matrix form: ")
    @time spinor_out_2 = gamma5_Dslash_wilson_matrix(spinor_in, lattice, 0.02)
    ddiff = 0.0
    for i in 1:lattice.ntot
        ddiff += spinor_out_1.s[i][1]-spinor_out_2.s[i][1]
        ddiff += spinor_out_1.s[i][2]-spinor_out_2.s[i][2]
    end
    if ddiff == 0
        println("CORRECTNESS PASSED: Two functions give same gamma_5*Dslash")
        return 0
    else
        println("CORRECTNESS FAILED: Two functions give different gamma_5*Dslash")
        return 1
    end
end
test_gamma5Dslash()