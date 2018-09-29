import Base: zero # Imported to be extended for REPL

"""
spinor = Spinor(ntot) will create a two-components spinor of lengt ntot
that are initilized to zero at all lattice points
"""
# Aliasing commonly used type
Field = Vector{Vector{ComplexF64}}
"""
Return field with all zeros. Required for IterativeSolvers
"""

function zero(x::Field)
    ntot = length(x)
    y = Field(undef, ntot)
    for i in 1:ntot
        y[i] = [0.0, 0.0]
    end
    return y
end

"""
Same as zero(x), but zero out the current array
"""
function zero!(x::Field)
    ntot = length(x)
    for i in 1:ntot
        x[i] = [0.0, 0.0]
    end
end

"""
Flatten out field spinor. Required for IterativeSolvers.
"""
function unravel(x::Field)
    ntot = length(x)
    y = Vector{ComplexF64}(undef, ntot*2)
    for i in 1:ntot
        for j in 1:2
            y[2*(i-1) + j] = x[i][j]
        end
    end
    return y
end
"""
Inverse of unravel. Required for IterativeSolvers.
"""
function ravel(x::Vector{ComplexF64})
    if length(x)%2 != 0
        error("length of x must be even")
    end
    ntot = convert(Int64, length(x)/2)
    y = Field(undef, ntot)
    for i in 1:ntot
        y[i] = x[2*i-1:2*i]
    end
    return y
end

"""
Construct Spinor object. If v is given, it will copy its value
and use it to construct field s
"""
mutable struct Spinor
    ntot::Int64
    s::Field # The components of spinor
    # Inner constructor method to initilize structure
    function Spinor(ntot::Int64, v=nothing)
        if v == nothing
            s = Field(undef, ntot)
            zero!(s)
        else
            s = deepcopy(v)
        end
        new(ntot, s)
    end
end

function test_spinor()
    nx = 5
    nt = 5
    spinor = Spinor(nx*nt)
    println("=======================================================================")
    println("=====                          Spinors                            =====")
    println("=======================================================================")
    println("Initilized zero spinor:")
    display(spinor.s)
    println("")
end
