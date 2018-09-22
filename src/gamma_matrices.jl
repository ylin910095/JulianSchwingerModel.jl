include("./spinor.jl")

if ((@isdefined gamma1) && (@isdefined gamma2) && (@isdefined gamma5))  == false
    # Conventions for gamma matrices
    const global gamma1 = [0.0 1.0; 1.0 0.0] # sigma_1
    const global gamma2 = [0.0 -1*im; im 0] # sigma_2
    const global gamma5 = [1.0 0.0; 0.0 -1.0] # sigma_3d
end

function gamma5mul!(field_in::Field)
    for i in 1:length(field_in)
        field_in[i] = gamma5 * field_in[i]
    end
end

function gamma5mul(field_in::Field)
    lin = length(field_in)
    field_out = Field(undef, lin)
    for i in 1:lin
        field_out[i] = gamma5 * field_in[i]
    end
    return field_out
end