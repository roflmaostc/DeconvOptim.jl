"""
    relative_energy_regain(ground_truth, rec)

Calculates the relative energy regain between the `ground_truth`
and the reconstruction.
Assumes that both arrays are 2 dimensional

"""
function relative_energy_regain(ground_truth, rec)
    # go to fourier space
    ground_truth_fft = fft(ground_truth)
    rec_fft = fft(rec)

    # a dict to store the values for certain frequencies
    # we store a list since some (rounded) frequencies occur more than once
    ΔE_R_dict = Dict{Float64, Vector{Float64}}()
    E_R_dict = Dict{Float64, Vector{Float64}}()

    # round the frequencies to 4 digits, alternative would be to bin
    round4(x) = round(x, digits=3)
    
    
    # iterate over the frequencies and calculate the relative energy regain
    for (i₂, f₂) in enumerate(fftfreq(size(rec_fft, 2)))
        for (i₁, f₁) in enumerate(fftfreq(size(rec_fft, 1)))
            f_res = round4(√(f₁^2 + f₂^2))
            Δ_E_R = abs2(ground_truth_fft[i₁, i₂] - rec_fft[i₁, i₂]) 
            E_R = abs2(ground_truth_fft[i₁, i₂]) 

            update_dict_list!(ΔE_R_dict, f_res, Δ_E_R)
            update_dict_list!(E_R_dict, f_res, E_R)
        end
    end
    
    
    # finally transform everything into a list of frequencies and 
    # a list of relative energy regains
    freqs = Float64[]
    G_R_list = Float64[]
    for f in sort(Float64.(keys(ΔE_R_dict)))
        push!(freqs, f)
        mean_ΔE_r = mean(ΔE_R_dict[f])
        mean_E_r = mean(E_R_dict[f])

        push!(G_R_list, (mean_E_r - mean_ΔE_r) / mean_E_r)
    end
    
    return freqs, G_R_list
end




"""
    update_dict_list!(d, k, v)

Updates the dict `d` which stores a list.
If `k` is in the keys of `d` we simply push `v` to the list
otherwise create a new list `[v]`
"""
function update_dict_list!(d, k, v)
    if haskey(d, k)
        push!(d[k], v)
    else
        d[k] = [v]
    end
    return d
end


"""
    normalized_cross_correlation(ground_truth, measured)

Calculates the normalized cross correlation.

External links: 
* [Wikipedia](https://en.wikipedia.org/wiki/Sombrero_function)
* [StatsBase.jl](https://juliastats.org/StatsBase.jl/stable/signalcorr/#StatsBase.crosscor)
"""
function normalized_cross_correlation(ground_truth, measured)
    fl(x) = collect(Iterators.flatten(x))
    ground_truth = fl(ground_truth)
    measured = fl(measured)

    ncc = crosscor(ground_truth, measured, [0], demean=false)
    return ncc
end
