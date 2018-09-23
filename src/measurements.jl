include("./lattice.jl")
include("./spinor.jl")
"""
2D plaq plaquette of unit size at site i
"""
function plaq(i::Int64, lattice::Lattice)
    return lattice.linkx[i] * lattice.linkt[lattice.rightx[i]] *
           conj(lattice.linkx[lattice.upt[i]]) *
           conj(lattice.linkt[i])
end

function measure_wilsonloop(lattice::Lattice)
    plaqtot = 0
    for i in 1:lattice.ntot
        plaqtot += plaq(i, lattice)
    end
    return plaqtot / lattice.ntot
end