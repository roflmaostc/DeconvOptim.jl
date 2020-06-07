export generate_psf, conv_real_otf, conv_real

using FFTW

export conv_real_otf


function conv_real_otf(rec, otf)
    return real(irfft(rfft(rec) .* otf, size(rec)[1]))
end

function conv_real_otf5D(rec, otf)
    return real(irfft(rfft(rec, [1, 2, 3]) .* otf, size(rec)[1], [1, 2, 3]))
end

function conv_real(img, psf)
    return real(irfft(rfft(rec, [1, 2, 3]) .* rfft(psf, [1, 2, 3]),
                      size(rec)[1], [1, 2, 3]))
end

function conv_real_otf_p(P, P_inv, rec, otf)
    return real(P_inv * ((P * rec) .* otf))
end


function rr(img)
    s = size(img)
    rarr = similar(img)
    for i = 1:s[1]
        for j = 1:s[2]
            rarr[i, j] = sqrt( (i-s[1] / 2)^2 + (j-s[2] / 2)^2)
        end
    end
    return rarr
end

function generate_psf(img, r)
    mask = rr(img) .< r
    mask_ft = ifft(ifftshift(mask))
    psf = abs2.(mask_ft)
    return psf ./ sum(psf)
end
