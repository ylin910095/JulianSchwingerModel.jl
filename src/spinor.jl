import Base: zero # Imported to be extended for REPL
FlatField = Vector{ComplexF64}
Field = Array{ComplexF64}

"""
Return field with all zeros in place. Required for IterativeSolvers.
"""
function zero!(x::Field)
    tlen = length(x)
    for i in 1:tlen
        x[i] = 0.0 + im * 0.0
    end
end

"""
Same as zero!, but not in place.
"""
function zero(x::Field)
    y = deepcopy(x)
    zero!(x)
    return x
end

"""
First component of Dirac index at site i for Wilson-like fermion
when we flatten it
"""
function dirac_comp1(i::Int64)
    return 2*(i-1) + 1
end

"""
Seonc component of Dirac index at site i for Wilson-like fermion
when we flatten it
"""
function dirac_comp2(i::Int64)
    return 2*(i-1) + 2
end
