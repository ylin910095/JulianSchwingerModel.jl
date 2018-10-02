include("./lattice.jl")
include("./solvers.jl")
include("./hmc_types.jl")
"""
S_gauge = sum_i beta(1 - Re U_{plaq})
Caluate the gauge action contribution at lattice
site i.  The constant beta * 1 is ignored in action.
"""
function SG(i::Int64, lattice::Lattice)
    return - lattice.beta *
           cos(lattice.anglex[i] + lattice.anglet[lattice.rightx[i]] -
               lattice.anglex[lattice.upt[i]] - lattice.anglet[i])
end

"""
Gauge force for U(x=i, mu=1)
"""
function dSG1(i::Int64, lattice::Lattice)
    return - lattice.beta * (
           sin(lattice.anglex[lattice.downt[i]] +
               lattice.anglet[lattice.rightx[lattice.downt[i]]] -
               lattice.anglex[i] - lattice.anglet[lattice.downt[i]]) -
           sin(lattice.anglex[i] + lattice.anglet[lattice.rightx[i]] -
               lattice.anglex[lattice.upt[i]] - lattice.anglet[i])
            )
end

"""
Gauge force for U(x=i, mu=2)
"""
function dSG2(i::Int64, lattice::Lattice)
    return lattice.beta * (
            sin(lattice.anglex[lattice.leftx[i]] + lattice.anglet[i] -
                lattice.anglex[lattice.leftx[lattice.upt[i]]] -
                lattice.anglet[lattice.leftx[i]]) -
            sin(lattice.anglex[i] + lattice.anglet[lattice.rightx[i]] -
                lattice.anglex[lattice.upt[i]] - lattice.anglet[i])
            )
end

"""
Internal function used in pforce1 and pforce2. Calculate the left-hand-side of
dot product that is independent of mu.

CG inversion takes place here. Q is returned by gamma5_Dslash_linearmap so
we dont have to reconstruct it everytime we call
"""
function pforce_common(Q::Any, pf::PseudoFermion, lattice::Lattice)
    # First invert pf
    psi = minres_Q(Q, lattice, lattice.mass, gamma5mul(pf.pf))
    Dm1_gamma5_psi = minres_Q(Q, lattice, lattice.mass, psi)
    return gamma5mul(Dm1_gamma5_psi), psi
end

"""
Fermion forces at site i, mu=1. Using equation (2.81) of Luscher 2010, ``Computational
Strategies in Lattice QCD``. lhs is the output from pforce_common.
"""
function pforce1(i::Int64, pf::PseudoFermion, lattice::Lattice, psi::FlatField,
                 lhs::FlatField)
    # Apply the variant of S_pf with respect of gaugel link
    left1_i = lattice.leftx[i]
    right1_i = lattice.rightx[i]
    link1 = lattice.linkx[i]

    # Implementing periodic bc in space
    # First term in dot product
    dotx = 0.5 * (
            conj(lhs[dirac_comp1(right1_i)]) * conj(link1) *
            (psi[dirac_comp1(i)] + psi[dirac_comp2(i)]) -
            conj(lhs[dirac_comp1(i)]) * link1 *
            (psi[dirac_comp1(right1_i)] - psi[dirac_comp2(right1_i)])
            )

    # Second term in dot product
    dott = 0.5 * (
            conj(lhs[dirac_comp2(right1_i)]) * conj(link1) *
            (psi[dirac_comp1(i)] + psi[dirac_comp2(i)]) -
            conj(lhs[dirac_comp2(i)]) * link1 *
            (-psi[dirac_comp1(right1_i)] + psi[dirac_comp2(right1_i)])
            )

    return -2*real(im*(dotx + dott))
end

"""
Fermion forces at site i, mu=2. Using equation (2.81) of Luscher 2010, ``Computational
Strategies in Lattice QCD``. It is important to have boundary condition consistent with
the Dslash operator. lhs is the output from pforce_common.
psi = D^{-1}phi where phi is pf field.
"""
function pforce2(i::Int64, pf::PseudoFermion, lattice::Lattice, psi::FlatField,
                 lhs::FlatField)
    # Apply the variant of S_pf with respect of gaugel link
    right2_i = lattice.upt[i]
    # Implementing antiperiodic bc in time
    if lattice.corr_indx[right2_i][2] == 1
        bterm = -1
    else
        bterm = 1
    end
    psi_right2_1i = bterm * psi[dirac_comp1(right2_i)]
    psi_right2_2i = bterm * psi[dirac_comp2(right2_i)]
    lhs_right2_1i = bterm * lhs[dirac_comp1(right2_i)]
    lhs_right2_2i = bterm * lhs[dirac_comp2(right2_i)]
    link2 = lattice.linkt[i]

    # First term in dot product
    dotx = 0.5 * (
            conj(lhs_right2_1i) * conj(link2)*
            (psi[dirac_comp1(i)] - im * psi[dirac_comp2(i)]) -
            conj(lhs[dirac_comp1(i)]) * link2 *
            (psi_right2_1i + im * psi_right2_2i)
            )
    # Second term in dot product
    dott = 0.5 * (
            conj(lhs_right2_2i) * conj(link2) *
            (im * psi[dirac_comp1(i)] + psi[dirac_comp2(i)]) -
            conj(lhs[dirac_comp2(i)]) * link2 *
            (-im * psi_right2_1i + psi_right2_2i)
            )
    return -2*real(im*(dotx + dott))
end
