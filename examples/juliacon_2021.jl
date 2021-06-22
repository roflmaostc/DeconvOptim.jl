### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 0507f8ed-8b64-48af-a6c4-ff7c4211b9e9
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 3310d2f7-450a-4cd1-9c3a-46d02d23a7c6
using Revise

# ╔═╡ d27b2d72-d264-11eb-0be5-13dcacfd2adc
using DeconvOptim, TestImages, ImageShow, Plots, LinearAlgebra, IndexFunArrays, Noise, FourierTools, SpecialFunctions, FFTW, LaTeXStrings, PlutoUI, Images, Tullio

# ╔═╡ 952b251c-207b-4412-b6e1-268fce1647d9
begin
	img = Float32.(testimage("fabio_gray"));
	img_1D = img[:, 200]
end;

# ╔═╡ fa8cd9c9-a2fd-495c-8d22-ada7bb9c39f6
otf(x, Δx=1) = begin
	x = abs(x)
	if x <= Δx 
		SpecialFunctions.jinc(x * Δx *(1-x/Δx)) .* 2 / π * (acos(x/Δx) - x/Δx * sqrt(1-(x/Δx)^2))
	else
		zero(x)
	end
end

# ╔═╡ 9a07bc88-be76-4531-bc91-df0d20c3221c
begin
	x = range(-1.5, 1.5, length=size(img, 1))
	freqs = fftshift(fftfreq(size(img_1D, 1), 1))
	psf = DeconvOptim.generate_psf(size(img), 30)
	psf_1D = psf[1, :]
	psf_1D ./= sum(psf_1D)
	otf_1D = abs.(ffts(psf_1D))
end;

# ╔═╡ 24fc86da-0b7c-493c-8977-2a19ef6dc133
img_n = poisson(DeconvOptim.conv(img, psf), 1000);

# ╔═╡ 3bda922d-552b-42ec-9055-33a141b5841a
blur(x, otf=otf_1D) = iffts(ffts(x) .* otf)

# ╔═╡ 061faf49-662c-4508-83b4-ddcf0970ed0d
md"### DeconvOptim.jl: Microscopy Image Deconvolution
"

# ╔═╡ 23829201-9756-4ef8-90c9-3917b761fe4b
load("../docs/src/assets/logo.png")

# ╔═╡ 30f21bb8-6d09-4fce-9d2a-568bfaf3ff7a
md"
* **Felix Wechsler:** Master Student at the IPHT Jena, Germany
* https://github.com/roflmaostc/DeconvOptim.jl
* `]add DeconvOptim`
"

# ╔═╡ 7a0a44c9-fb07-44e6-9a8b-8f720b84e6f6
md"""### Image Convolution

* Typical description of isotropic blur of an image
* The blurring kernel describes blur
    * In optics/microscopy a finite sized dot called Point Spread Function (**PSF**)
    * often a Gaussian function used in image processing
    * cigarre shaped object for motion blur

Mathematically:

$(S * \text{PSF})(\mathbf r) = \int_{-\infty}^{\infty} S(\mathbf r - \mathbf x) \cdot \text{PSF}(\mathbf x) \, \mathrm d \mathbf x$

Discrete version:

$(S * \text{PSF})[i, j] = \sum_{m,n} S[i-m, j-n] \cdot \text{PSF}[m, n]$
"""

# ╔═╡ 49686d9a-1683-428f-83ba-a9131c2ad432
[Gray.(img) Gray.(DeconvOptim.conv(img, psf))]

# ╔═╡ 491ec7cb-1663-4acf-b81f-9acafba2b63d
md"## Convolution Theorem


* For small kernels where the kernel has $K$ datapoints, it takes $\mathcal O(K \cdot N)$
* For large kernels naive integral calculation takes $\mathcal O(N^2)$ operations for a 1D dataset with $N$ points 

* Using the Convolution theorem
$(S * \text{PSF})(\mathbf r) = \mathcal{F}^{-1}\bigg[ \mathcal{F}[S] \cdot \mathcal{F}[\text{PSF}]  \bigg]$

* we can express the convolution with a Fast Fourier Transform (FFT) which only takes $\mathcal O(N \log(N))$ operations


*  $\mathcal{F}[\text{PSF}]$ is called the $\text{OTF}$


"

# ╔═╡ b6f9e42e-5a23-40ad-9e73-8f02da48f69c
md"### Optical System act as low pass filter"

# ╔═╡ e31f4704-5bce-4c8d-b3a7-873a3460d9f8
plot(x, otf.(x), xlabel="frequency / maximum frequency", ylabel="contrast")

# ╔═╡ 1a6cd3d6-20d4-4597-bd08-424d6c8e3df7
md"## Frequency spectrum of a sample"

# ╔═╡ 83947a1d-37a5-4bb8-8438-20b1b8eb0dc0
plot(freqs, abs.(ffts(img_1D)), ylabel="real part of FFT output in AU", 
			xlabel="frequency in 1/px", yaxis=:log,
			ylims=(1e-4, 1e2))

# ╔═╡ dc8cf4e2-a87c-47ea-8247-3ae772852241
md"## Frequency spectrum of blurred sample $Y(\mathbf r)$

Blurred sample:
$Y(\mathbf r) = (S * \text{PSF})(\mathbf r)$
"

# ╔═╡ f8033d13-faee-41f5-bdd6-ae8721e8b8a8
begin
	plot(freqs, abs.(ffts(blur(img_1D))), yaxis=:log, ylabel="real part of FFT output in AU", xlabel="frequency in 1/px", ylims=(1e-4, 1e2), label="blurred")
	plot!(freqs, abs.(ffts(img_1D)), ylabel="real part of FFT output in AU", 
				xlabel="frequency in 1/px", yaxis=:log,
				ylims=(1e-4, 1e2), label="ground truth")
	#plot!(freqs, abs.(ffts(DeconvOptim.conv(img_1D, psf_1D))))
end

# ╔═╡ fe7e0292-31f6-43b5-83a5-a38698a87563
md"## Deconvolution Pipeline"

# ╔═╡ e9ef5ba4-56c0-4595-bc28-e04882f44a9a
load("../docs/src/assets/tex/pipeline.png")

# ╔═╡ 90e5708c-05a4-46e4-b1e4-9a61c96dae32
TV_by_hand(x) = @tullio r = sqrt(1f-8 + abs2(x[i, j] - x[i+1, j]) + 
						 	     abs2(x[i, j] - x[i, j+1]))

# ╔═╡ 03139ac5-3525-4fad-abf1-84421492b763
DeconvOptim.generate_TV(2, [1,2], [1,1], 1, 0)[1]

# ╔═╡ 8f556ec3-1d90-4f07-9099-047724a568d2


# ╔═╡ ffe29e8d-dad6-4238-a252-c8cff8f25613
TV_by_DeconvOptim = DeconvOptim.TV(num_dims=4, mode="forward")

# ╔═╡ 4e84e739-9c59-4939-8b04-aec7dc069d67
md"
### Deconvolve with DeconvOptim.jl

"

# ╔═╡ 15d54e5e-64f4-4a1d-8cc8-9334b2e3784f
md"$(@bind iter Slider(0:50, show_value=true))"

# ╔═╡ 88d83629-f1b2-4d69-928f-d4acfbc76b70
reg_1D = TV(num_dims=1);

# ╔═╡ 438a6639-bd35-464f-a81d-d98eb65e006e
res_1D, o = deconvolution(img_1D, psf_1D, iterations=iter, regularizer=reg_1D);

# ╔═╡ 5c8030b7-8805-4771-9329-23abb2744544
begin
	plot(freqs, abs.(ffts(blur(img_1D))), yaxis=:log, ylabel="real part of FFT output in AU", xlabel="frequency in 1/px", ylims=(1e-4, 1e2), label="blurred")
	plot!(freqs, abs.(ffts(img_1D)), ylabel="real part of FFT output in AU", 
				xlabel="frequency in 1/px", yaxis=:log,
				ylims=(1e-4, 1e2), label="ground truth")
	plot!(freqs, abs.(ffts(res_1D)), label="deconvolved image")
end

# ╔═╡ 0ce13c2d-c043-4cdc-bd0b-051890e6f768
begin
	res_TV, o_img1 = deconvolution(img_n, psf, regularizer=TV(), iterations=30, λ=0.02f0)
	res_Tik, o_img2 = deconvolution(img_n, psf, regularizer=DeconvOptim.Tikhonov(mode="spatial_grad_square"), iterations=30, λ=0.03f0)
end;

# ╔═╡ fc93fa3d-8599-463b-b9e6-8043b90e9d63
[Gray.(img_n) Gray.(res_TV) Gray.(res_Tik)]

# ╔═╡ Cell order:
# ╠═3310d2f7-450a-4cd1-9c3a-46d02d23a7c6
# ╠═0507f8ed-8b64-48af-a6c4-ff7c4211b9e9
# ╠═d27b2d72-d264-11eb-0be5-13dcacfd2adc
# ╠═952b251c-207b-4412-b6e1-268fce1647d9
# ╠═24fc86da-0b7c-493c-8977-2a19ef6dc133
# ╠═3bda922d-552b-42ec-9055-33a141b5841a
# ╟─fa8cd9c9-a2fd-495c-8d22-ada7bb9c39f6
# ╠═9a07bc88-be76-4531-bc91-df0d20c3221c
# ╟─061faf49-662c-4508-83b4-ddcf0970ed0d
# ╟─23829201-9756-4ef8-90c9-3917b761fe4b
# ╟─30f21bb8-6d09-4fce-9d2a-568bfaf3ff7a
# ╠═7a0a44c9-fb07-44e6-9a8b-8f720b84e6f6
# ╟─49686d9a-1683-428f-83ba-a9131c2ad432
# ╠═491ec7cb-1663-4acf-b81f-9acafba2b63d
# ╟─b6f9e42e-5a23-40ad-9e73-8f02da48f69c
# ╟─e31f4704-5bce-4c8d-b3a7-873a3460d9f8
# ╟─1a6cd3d6-20d4-4597-bd08-424d6c8e3df7
# ╟─83947a1d-37a5-4bb8-8438-20b1b8eb0dc0
# ╟─dc8cf4e2-a87c-47ea-8247-3ae772852241
# ╠═f8033d13-faee-41f5-bdd6-ae8721e8b8a8
# ╟─fe7e0292-31f6-43b5-83a5-a38698a87563
# ╟─e9ef5ba4-56c0-4595-bc28-e04882f44a9a
# ╠═90e5708c-05a4-46e4-b1e4-9a61c96dae32
# ╠═03139ac5-3525-4fad-abf1-84421492b763
# ╠═8f556ec3-1d90-4f07-9099-047724a568d2
# ╠═ffe29e8d-dad6-4238-a252-c8cff8f25613
# ╟─4e84e739-9c59-4939-8b04-aec7dc069d67
# ╟─15d54e5e-64f4-4a1d-8cc8-9334b2e3784f
# ╟─88d83629-f1b2-4d69-928f-d4acfbc76b70
# ╟─438a6639-bd35-464f-a81d-d98eb65e006e
# ╟─5c8030b7-8805-4771-9329-23abb2744544
# ╟─0ce13c2d-c043-4cdc-bd0b-051890e6f768
# ╠═fc93fa3d-8599-463b-b9e6-8043b90e9d63
