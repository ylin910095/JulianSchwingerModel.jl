include("./lattice.jl")

"""
S_gauge = \sum_i \beta(1 - Re U_{plaq})
Caluate the gauge action contribution at lattice
site i.  The constant \beta * 1 is ignored in action.
"""
function SG(i::Int64, lattice::Lattice)
    return - lattice.beta * 
           cos(lattice.linkx[i] + lattice.linkt[lattice.rightx[i]] -
               lattice.linkx[lattice.upt[i]] - lattice.linkt[i])
end

"""
Derivative of SG with respect to \theta_1(x) (gauge angle in x direction)
Used in HMC updating.
"""
function dSG1(i::Int64, lattice::Lattice)
    return - lattice.beta * (
           sin(lattice.linkx[lattice.downt[i]] + 
               lattice.linkt[lattice.rightx[lattice.downt[i]]] -
               lattice.linkx[i] - lattice.linkt[lattice.downt[i]]) +
           sin(lattice.linkx[i] + lattice.linkt[lattice.rightx[i]] -
               lattice.linkx[lattice.upt[i]] - lattice.linkt[i])
            )
end

"""
Derivative of SG with respect to \theta_2(x) (gauge angle in t direction)
Used in HMC updating
"""
function dSG2(i::Int64, lattice::Lattice}
    return lattice.beta * (
            sin(lattice.linkx[lattice.leftx[i]] + lattice.linkt[i] - 
                lattice.linkx[lattice.leftx[lattice.upt[i]]] -
                lattice.linkt[lattice.leftx[i]]) -
            sin(lattice.linkx[i] + lattice.linkt[lattice.rightx[i]] - 
                lattice.linkt[lattice.upt[i]] - lattice.linkt[i])
            )
end