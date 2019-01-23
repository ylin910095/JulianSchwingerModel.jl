if ((@isdefined gamma1) && (@isdefined gamma2) && (@isdefined gamma5))  == false
    # Conventions for gamma matrices
    const global gamma1 = [0.0 1.0; 1.0 0.0] # sigma_1
    const global gamma2 = [0.0 -1*im; im 0] # sigma_2
    const global gamma5 = [1.0 0.0; 0.0 -1.0] # sigma_3
end

"""
Left multiplication of gamma5 in place
"""
function gamma5mul!(field_in::Field)
    for i in 1:size(field_in)[2]
        field_in[:, i] = gamma5 * field_in[:, i]
    end
end
"""
Left multiplication of gamma5
"""
function gamma5mul(field_in::Field)
    y = deepcopy(field_in)
    gamma5mul!(y)
    return y
end

function gamma5mul!(ffield_in::FlatField)
    for i in 1:Int(length(ffield_in)/2)
        ffield_in[dirac_comp1(i)] = gamma5[1, 1] * ffield_in[dirac_comp1(i)] +
                                    gamma5[1, 2] * ffield_in[dirac_comp2(i)]
        ffield_in[dirac_comp2(i)] = gamma5[2, 1] * ffield_in[dirac_comp1(i)] +
                                    gamma5[2, 2] * ffield_in[dirac_comp2(i)]
    end
end

function gamma5mul(ffield_in::FlatField)
    y = deepcopy(ffield_in)
    gamma5mul!(y)
    return y
end
"""
function mulgamma5!(ffield_in::FlatField)
    for i in 1:Int(length(ffield_in)/2)
        ffield_in[dirac_comp1(i)] = gamma5[1, 1] * ffield_in[dirac_comp1(i)] +
                                    gamma5[1, 2] * ffield_in[dirac_comp2(i)]
        ffield_in[dirac_comp2(i)] = gamma5[2, 1] * ffield_in[dirac_comp1(i)] +
                                    gamma5[2, 2] * ffield_in[dirac_comp2(i)]
    end
end

function mulgamma5(ffield_in::FlatField)
    y = deepcopy(ffield_in)
    mulgamma5!(y)
    return y
end
"""

function mulgamma5(field_in::Field)
    lin = length(field_in)
    field_out = Field(undef, lin)
    for i in 1:lin
        field_out[i] = field_in[i] * gamma5
    end
end
