"""
    conv_aux(conv_otf, rec, otf)

Calculate the convolution between `rec` and `otf`.
The used convolution function is `conv_otf`.
`conv_otf` can be exchanged to be a rfft, fft or plan_fft based routine.

This function is just defined to speed up automatic differentiation
and it's custom defined adjoint.
"""
function conv_aux(conv_otf, rec, otf)
    return conv_otf(rec, otf)
end

 # define custom adjoint for conv_aux
function ChainRulesCore.rrule(::typeof(conv_aux), conv, rec, otf)
    Y = conv_aux(conv, rec, otf)
    function conv_aux_pullback(barx)
        return zero(eltype(rec)), zero(eltype(rec)), conv(barx, conj(otf)), zero(eltype(rec))
    end
    return Y, conv_aux_pullback
end
