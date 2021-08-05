"""
Q = gamma5_Dslash_wilson
Solve Qx = source using conjugate gradient
return FlatField
"""
function minres_Q(Q::Any, lattice::Lattice, mass::Float64,
                  source::FlatField, verbose::Bool=false)
    sol = minres(Q, source; reltol=1e-6, verbose=verbose)
    return sol
end