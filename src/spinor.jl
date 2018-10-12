import Base: zero # Imported to be extended for REPL


"""
Base type for spinor fields. FlatField is a vector of 
length 2*lattice.ntot where index 2*(i-1) + 1 is the first Dirac
component at lattice site i, and 2*(i-1) + 2 is the second Dirac
component at lattice site i.
"""
FlatField = Vector{ComplexF64}

"""
This is not compatible with IterativeSolvers.jl so not used
"""
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
@inline function dirac_comp1(i::Int64)
    return 2*(i-1) + 1
end

"""
Seonc component of Dirac index at site i for Wilson-like fermion
when we flatten it
"""
@inline function dirac_comp2(i::Int64)
    return 2*(i-1) + 2
end
