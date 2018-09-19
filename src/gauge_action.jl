include("./lattice.jl")

"""
S_gauge = \sum_i \beta(1 - Re U_{plaq})
Caluate the gauge action contribution at lattice
site i.  The constant \beta * 1 is ignored in action.
"""
function SG(i::Int64, lattice::Lattice)
    return - lattice.beta * 
           cos(lattice.anglex[i] + lattice.anglet[lattice.rightx[i]] -
               lattice.anglex[lattice.upt[i]] - lattice.anglet[i])
end

"""
Derivative of SG with respect to \theta_1(x) (gauge angle in x direction)
Used in HMC updating.
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
Derivative of SG with respect to \theta_2(x) (gauge angle in t direction)
Used in HMC updating.
"""
function dSG2(i::Int64, lattice::Lattice}
    return lattice.beta * (
            sin(lattice.anglex[lattice.leftx[i]] + lattice.anglet[i] - 
                lattice.anglex[lattice.leftx[lattice.upt[i]]] -
                lattice.anglet[lattice.leftx[i]]) -
            sin(lattice.anglex[i] + lattice.anglet[lattice.rightx[i]] - 
                lattice.anglet[lattice.upt[i]] - lattice.anglet[i])
            )
end