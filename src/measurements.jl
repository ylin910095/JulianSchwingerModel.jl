"""
2D plaq plaquette of unit size at site i
"""
@inline function plaq(i::Int64, lattice::Lattice)::Int64
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

"""
Project a variable of type Field into an Array with
shape (2, 2, lattice.nx, lattice.nt) so it can be passed
to TensorOperations.

field_in1 and field_in2 are the spacelike and timelike Dirac
component of the propagators.
"""
function projectfield(field_in1::FlatField, field_in2::FlatField,
                      lattice::Lattice)
    field_out = Array{ComplexF64}(undef, 2, 2, lattice.nx,
                                               lattice.nt)
    tensorfield_in1 = reshape(field_in1, 2, lattice.ntot)
    for i in 1:lattice.ntot
        x = lattice.corr_indx[i][1]
        t = lattice.corr_indx[i][2]
        field_out[1, 1, x, t] = field_in1[dirac_comp1(i)]
        field_out[2, 1, x, t] = field_in2[dirac_comp1(i)]
        field_out[1, 2, x, t] = field_in1[dirac_comp2(i)]
        field_out[2, 2, x, t] = field_in2[dirac_comp2(i)]
    end
    return field_out
end

"""
pion triplet correlation function for a given source
"""
function measure_pion(prop::Any, lattice::Lattice)
    anst = zero(Vector{Float64}(undef, lattice.nt))
    # Map it to multidimensional array instead of array
    # so it can be passed to TensorOperations
    proparray = projectfield(prop[1], prop[2], lattice)
    D = Array{ComplexF64}(undef, lattice.nt, lattice.nt)
    TensorOperations.@tensor begin
        D[t1, t2] = conj(proparray[a, i, b, t1]) * proparray[a, i, b, t2]
    end
    ans = [D[t,t] for t in 1:lattice.nt]
    return ans
end

function measure_a0(prop::Any, lattice::Lattice)
    anst = zero(Vector{Float64}(undef, lattice.nt))
    # Map it to multidimensional array instead of array
    # so it can be passed to TensorOperations
    proparray = projectfield(prop[1], prop[2], lattice)
    D = Array{ComplexF64}(undef, lattice.nt, lattice.nt)
    TensorOperations.@tensor begin
        D[t1, t2] = gamma5[a, c] * conj(proparray[g, c, x, t1]) *
                    gamma5[g, b] * proparray[b, a, x, t2]
    end
    ans = [D[t,t] for t in 1:lattice.nt]
    return ans
end

function measure_g1(prop::Any, lattice::Lattice)
    anst = zero(Vector{Float64}(undef, lattice.nt))
    # Map it to multidimensional array instead of array
    # so it can be passed to TensorOperations
    proparray = projectfield(prop[1], prop[2], lattice)
    D = Array{ComplexF64}(undef, lattice.nt, lattice.nt)
    TensorOperations.@tensor begin
        D[t1, t2] = gamma1[a,b] * gamma5[b, f] *
                    conj(proparray[m, f, t0, t1]) *
                    gamma5[m,c] * gamma1[c, d] * proparray[d, a, t0, t2]
    end
    ans = [D[t,t] for t in 1:lattice.nt]
    return ans
end

