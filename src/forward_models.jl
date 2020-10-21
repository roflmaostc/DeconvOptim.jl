export conv_aux

function conv_aux(conv, rec, otf)
    return conv(rec, otf)
end


function ChainRulesCore.rrule(::typeof(conv_aux), conv, rec, otf)
    Y = conv_aux(conv, rec, otf)
    function conv_aux_pullback(barx)
        return NO_FIELDS, DoesNotExist(), conv(barx, conj(otf)), DoesNotExist()
    end
    return Y, conv_aux_pullback
end
#= @adjoint conv_aux(conv, rec, otf) = begin =#
#=     ∇ = (c -> (-1, conv(c, conj(otf)), -1)) =#
#=     value = conv_aux(conv, rec, otf) =# 
#=     return (value, ∇) =#
#= end =#
