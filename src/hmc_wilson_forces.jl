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
               lattice.anglex[i] - lattice.anglet[lattice.downt[i]]) +
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

CG inversion takes place here.
"""
function pforce_common(pf::PseudoFermion, lattice::Lattice)
    # First invert pf
    x0 = Field(undef, lattice.ntot) # Starting guess
    zero!(x0) # Zero initial guess
    Dm1_gamma5_psi = cg_Q(lattice, lattice.mass, x0, pf.Dm1pf.s)
    gamma5mul!(Dm1_gamma5_psi)
    return Dm1_gamma5_psi
end

"""
Fermion forces at site i, mu=1. Using equation (2.81) of Luscher 2010, ``Computational 
Strategies in Lattice QCD``. lhs is the output from pforce_common.
"""
function pforce1(i::Int64, pf::PseudoFermion, lattice::Lattice, lhs::Field)

    # Calculate the factor on the right-hand-side of the dot product
    # psi = D^{-1}\phi where \phi is pf field.
    psi = pf.Dm1pf.s

    # Apply the variant of S_pf with respect of gaugel link
    left1_i = lattice.leftx[i]
    right1_i = lattice.rightx[i]
    link1 =lattice.linkx[i]

    # Implementing periodic bc in space
    # First term in dot product
    dotx = 0.5*(
            conj(lhs[right1_i][1]) * conj(link1)*(psi[i][1] + psi[i][2]) - 
            conj(lhs[i][1]) * link1 * (psi[right1_i][1] - psi[right1_i][2]) 
            )
    # Second term in dot product
    dott = 0.5 * (
            conj(lhs[right1_i][2]) * conj(link1)*(psi[i][1] + psi[i][2]) - 
            conj(lhs[i][2]) * link1 * (-psi[right1_i][1] + psi[right1_i][2]) 
            )
    return -2*real(dotx + dott)
end

"""
Fermion forces at site i, mu=2. Using equation (2.81) of Luscher 2010, ``Computational 
Strategies in Lattice QCD``. It is important to have boundary condition consistent with
the Dslash operator. lhs is the output from pforce_common.
"""
function pforce2(i::Int64, pf::PseudoFermion, lattice::Lattice, lhs::Field)

    # Calculate the factor on the right-hand-side of the dot product
    # psi = D^{-1}\phi where \phi is pf field.
    psi = pf.Dm1pf.s

    # Apply the variant of S_pf with respect of gaugel link
    right2_i = lattice.upt[i]
    # Implementing antiperiodic bc in time
    if lattice.corr_indx[right2_i][2] == 1
        psi_right2_i = -psi[right2_i]
        lhs_right2_i = -lhs[right2_i]
    else
        psi_right2_i = psi[right2_i]
        lhs_right2_i = lhs[right2_i]
    end
    link2 = lattice.linkt[i]

    # Implementing periodic bc in space
    # First term in dot product
    dotx = 0.5*(
            conj(lhs_right2_i[1]) * conj(link2)*(psi[i][1] - im * psi[i][2]) - 
            conj(lhs[i][1]) * link2 *(psi[right2_i][1] + im * psi_right2_i[2]) 
            )
    # Second term in dot product
    dott = 0.5 * (
            conj(lhs_right2_i[2]) * conj(link2)*(im * psi[i][1] + psi[i][2]) - 
            conj(lhs[i][2]) * link2 *(-im * psi[right2_i][1] + psi_right2_i[2]) 
            )
    return -2*real(dotx + dott)
end