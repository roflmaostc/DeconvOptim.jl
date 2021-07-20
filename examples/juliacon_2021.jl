### A Pluto.jl notebook ###
# v0.14.5

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
	psf = Float32.(DeconvOptim.generate_psf(size(img), 20))
	psf_1D = psf[1, :]
	psf_1D ./= sum(psf_1D)
	otf_1D = abs.(ffts(psf_1D))
end;

# ╔═╡ 24fc86da-0b7c-493c-8977-2a19ef6dc133
img_n = Float32.(poisson(DeconvOptim.conv(img, psf), 1000));

# ╔═╡ 3bda922d-552b-42ec-9055-33a141b5841a
blur(x, otf=otf_1D) = iffts(ffts(x) .* otf)

# ╔═╡ 308e2562-5a20-4082-8a6d-9fb135738c49
md"
Mathematically:

$(S * \text{PSF})(\mathbf r) = \int_{-\infty}^{\infty} S(\mathbf r - \mathbf x) \cdot \text{PSF}(\mathbf x) \, \mathrm d \mathbf x$
"

# ╔═╡ b542187a-3ae7-4430-a81d-f968d9000427
img_blurry = DeconvOptim.conv(img, psf);

# ╔═╡ 88d83629-f1b2-4d69-928f-d4acfbc76b70
reg_1D = TV(num_dims=1);

# ╔═╡ d6d0f436-48fd-4e8a-863c-3e9d3850cc91
begin
	reg_tik = Tikhonov()
	reg_TV = TV()
	reg_GR = DeconvOptim.GR()
end

# ╔═╡ 061faf49-662c-4508-83b4-ddcf0970ed0d
md"### DeconvOptim.jl: Microscopy Image Deconvolution
"

# ╔═╡ 23829201-9756-4ef8-90c9-3917b761fe4b
load("../docs/src/assets/logo.png")

# ╔═╡ 30f21bb8-6d09-4fce-9d2a-568bfaf3ff7a
md"
* **Felix Wechsler:** Master Student at the Leibniz Institute of Photonic Technology in  Jena, Germany
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


Discrete version:

$(S * \text{PSF})[i] = \sum_{m} S[i-m] \cdot \text{PSF}[m]$
"""

# ╔═╡ 49686d9a-1683-428f-83ba-a9131c2ad432
[Gray.(img) Gray.(DeconvOptim.conv(img, psf))]

# ╔═╡ 491ec7cb-1663-4acf-b81f-9acafba2b63d
md"## Convolution Theorem

$(S * \text{PSF})(\mathbf r) = \mathcal{F}^{-1}\bigg[ \mathcal{F}[S] \cdot \mathcal{F}[\text{PSF}]  \bigg]$

* we can express the convolution with a Fast Fourier Transform (FFT) which only takes $\mathcal O(N \log(N))$ operations


* For large kernels (especially in 3D), sliding kernels are slower


*  $\mathcal{F}[\text{PSF}]$ is called the $\text{OTF}$

"

# ╔═╡ b6f9e42e-5a23-40ad-9e73-8f02da48f69c
md"### Optical System act as low pass filter

*  $\text{OTF}$ shows the frequency throughput

"

# ╔═╡ e31f4704-5bce-4c8d-b3a7-873a3460d9f8
plot(x, otf.(x), xlabel="frequency / maximum frequency", ylabel="contrast")

# ╔═╡ dc8cf4e2-a87c-47ea-8247-3ae772852241
md"## Frequency spectrum of blurred sample $Y(\mathbf r)$

Blurred sample:
$Y(\mathbf r) = (S * \text{PSF})(\mathbf r)$
"

# ╔═╡ f8033d13-faee-41f5-bdd6-ae8721e8b8a8
begin
	plot(freqs, abs.(ffts(img_blurry)[:, 128]), yaxis=:log, ylabel="real part of FFT output in AU", xlabel="frequency in 1/px", ylims=(1e-4, 1e2), label="blurred")
	plot!(freqs, abs.(ffts(img)[:, 128]), ylabel="abs of FFT output in AU", 
				xlabel="frequency in 1/px", yaxis=:log,
				ylims=(1e-4, 1e4), label="ground truth")
	#plot!(freqs, abs.(ffts(DeconvOptim.conv(img_1D, psf_1D))))
end

# ╔═╡ fe7e0292-31f6-43b5-83a5-a38698a87563
md"## Deconvolution Pipeline

* based on:
    * Zygote.jl
    * Optim.jl
    * Tullio.jl
    * CUDA.jl
"

# ╔═╡ e9ef5ba4-56c0-4595-bc28-e04882f44a9a
load("../docs/src/assets/tex/pipeline.png")

# ╔═╡ 90e5708c-05a4-46e4-b1e4-9a61c96dae32
TV_by_hand(x) = @tullio r = sqrt(1f-8 + abs2(x[i, j] - x[i+1, j]) + 
						 	     abs2(x[i, j] - x[i, j+1]))

# ╔═╡ 03139ac5-3525-4fad-abf1-84421492b763
DeconvOptim.generate_TV(4, [1,2, 3], [1,1, 1], 1, 0)[1]

# ╔═╡ 4e84e739-9c59-4939-8b04-aec7dc069d67
md"
### Deconvolve with DeconvOptim.jl

"

# ╔═╡ 9d9a5da2-14df-46e6-b7e6-5a33aade1754
@bind reg_list2 Select(["1" => ("Tikhonov"), "2" => ("Total Variation TV"), "3" => ("Good's Roughness GR")])

# ╔═╡ 5ae1a6a3-4505-4123-9f1e-8a1d4ac0b4e1
reg = [reg_tik, reg_TV, reg_GR][parse(Int, reg_list2)]

# ╔═╡ 15d54e5e-64f4-4a1d-8cc8-9334b2e3784f
md"
iterations = 
$(@bind iter Slider(0:50, show_value=true))

λ = $(@bind λ Slider(0:0.001:0.3, show_value=true))

regularizer = $(@bind reg_bool CheckBox())"

# ╔═╡ 803368e6-53fd-4413-b3f5-ffe46ee8983e
img_deconv, res_img = deconvolution(img_blurry, psf, regularizer=reg_bool ? reg : nothing, iterations=iter, λ=λ);

# ╔═╡ e58e1f63-2c81-48fb-866a-4bb70bd428a6
Gray.(img_deconv)

# ╔═╡ 438a6639-bd35-464f-a81d-d98eb65e006e
res_1D, o = deconvolution(real(blur(img_1D)), psf_1D, iterations=iter, regularizer=reg_1D, λ=0.01);

# ╔═╡ 7a16df73-95ad-47f5-907c-6fd23c6000cf
[Gray.(img) Gray.(img_blurry) Gray.(img_deconv)]

# ╔═╡ 5c8030b7-8805-4771-9329-23abb2744544
begin
	plot(freqs, abs.(ffts(img_blurry)[:, 128]), yaxis=:log, ylabel="abs of FFT output in AU", xlabel="frequency in 1/px", ylims=(1e-4, 1e2), label="blurred")
	plot!(freqs, abs.(ffts(img)[:, 128]),  yaxis=:log,
				ylims=(1e-4, 1e4), label="ground truth")
	plot!(freqs, abs.(ffts(img_deconv)[:, 128]), label="deconvolved image")
end

# ╔═╡ ff9af79c-f06c-42ac-9c8d-6f09d2ff4056
[Gray.(img_1D); Gray.(res_1D)];

# ╔═╡ d8aee845-e922-41da-a17e-37ffa3e692f0
md"### Real Microscopy Data"

# ╔═╡ fc93fa3d-8599-463b-b9e6-8043b90e9d63
load("figures/real_data_large.png")

# ╔═╡ 97c45bfd-d1ef-49ad-908d-7360c03b0170
md"Image taken from [DeconvolutionLab2](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)."

# ╔═╡ bf37664a-f726-46b6-9592-da419165af91
md"## Conclusion - DeconvOptim.jl
"

# ╔═╡ f8117250-bac8-43a1-aae4-5c8bab3a522d
[Gray.(ones(130, 012)) load("../docs/src/assets/logo.png")]

# ╔═╡ b5a70276-e9b9-46e8-8c67-0cad2cfa19da
md"* Flexible Image Deconvolution Software
* N-dimensional signal deconvolution
* Works both on CPU and GPUs
    * GPUs usually 5-15x speed improvement
"

# ╔═╡ Cell order:
# ╠═3310d2f7-450a-4cd1-9c3a-46d02d23a7c6
# ╠═0507f8ed-8b64-48af-a6c4-ff7c4211b9e9
# ╠═d27b2d72-d264-11eb-0be5-13dcacfd2adc
# ╠═952b251c-207b-4412-b6e1-268fce1647d9
# ╠═24fc86da-0b7c-493c-8977-2a19ef6dc133
# ╠═3bda922d-552b-42ec-9055-33a141b5841a
# ╟─fa8cd9c9-a2fd-495c-8d22-ada7bb9c39f6
# ╠═9a07bc88-be76-4531-bc91-df0d20c3221c
# ╟─308e2562-5a20-4082-8a6d-9fb135738c49
# ╠═b542187a-3ae7-4430-a81d-f968d9000427
# ╠═88d83629-f1b2-4d69-928f-d4acfbc76b70
# ╠═e58e1f63-2c81-48fb-866a-4bb70bd428a6
# ╠═d6d0f436-48fd-4e8a-863c-3e9d3850cc91
# ╠═803368e6-53fd-4413-b3f5-ffe46ee8983e
# ╠═5ae1a6a3-4505-4123-9f1e-8a1d4ac0b4e1
# ╠═438a6639-bd35-464f-a81d-d98eb65e006e
# ╟─061faf49-662c-4508-83b4-ddcf0970ed0d
# ╟─23829201-9756-4ef8-90c9-3917b761fe4b
# ╟─30f21bb8-6d09-4fce-9d2a-568bfaf3ff7a
# ╟─7a0a44c9-fb07-44e6-9a8b-8f720b84e6f6
# ╟─49686d9a-1683-428f-83ba-a9131c2ad432
# ╟─491ec7cb-1663-4acf-b81f-9acafba2b63d
# ╟─b6f9e42e-5a23-40ad-9e73-8f02da48f69c
# ╟─e31f4704-5bce-4c8d-b3a7-873a3460d9f8
# ╟─dc8cf4e2-a87c-47ea-8247-3ae772852241
# ╟─f8033d13-faee-41f5-bdd6-ae8721e8b8a8
# ╟─fe7e0292-31f6-43b5-83a5-a38698a87563
# ╟─e9ef5ba4-56c0-4595-bc28-e04882f44a9a
# ╠═90e5708c-05a4-46e4-b1e4-9a61c96dae32
# ╠═03139ac5-3525-4fad-abf1-84421492b763
# ╟─4e84e739-9c59-4939-8b04-aec7dc069d67
# ╟─9d9a5da2-14df-46e6-b7e6-5a33aade1754
# ╟─15d54e5e-64f4-4a1d-8cc8-9334b2e3784f
# ╟─7a16df73-95ad-47f5-907c-6fd23c6000cf
# ╟─5c8030b7-8805-4771-9329-23abb2744544
# ╟─ff9af79c-f06c-42ac-9c8d-6f09d2ff4056
# ╟─d8aee845-e922-41da-a17e-37ffa3e692f0
# ╟─fc93fa3d-8599-463b-b9e6-8043b90e9d63
# ╟─97c45bfd-d1ef-49ad-908d-7360c03b0170
# ╟─bf37664a-f726-46b6-9592-da419165af91
# ╟─f8117250-bac8-43a1-aae4-5c8bab3a522d
# ╟─b5a70276-e9b9-46e8-8c67-0cad2cfa19da
