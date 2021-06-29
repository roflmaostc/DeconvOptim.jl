### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 0552a944-cdbe-11eb-2e63-49f12bee1624
using FourierTools, TestImages, Colors, ImageShow, ImageCore, Statistics

# ╔═╡ 9a73fe7d-d298-4791-8f29-5a962d1242d1
gauss(y, x, σ=1) = 1 / σ / √(2π) * exp(-0.5 * (x^2 + y^2) / σ^2)

# ╔═╡ b8b2b314-4fed-420e-9061-6c2716c134c8
begin
	y = fftpos(512, 512)
	x = y'
end;

# ╔═╡ 97dc4b9d-0b32-4057-8a22-85cb40ae1ebf
begin
	kernel = ifftshift_view(gauss.(y, x, Ref(3)))
	kernel ./= sum(kernel)
end;

# ╔═╡ 7500e8d2-16a7-4ea3-9192-78b4acfa327e
function conv_pad(A, B, value=zero(eltype(A)), pad=10)
	A_1 = value .+ FourierTools.select_region(A .- value, new_size=size(A) .+ pad)
	B_1 = ifftshift_view(FourierTools.select_region(fftshift_view(B), new_size=size(A) .+ pad))
	return FourierTools.select_region(conv(A_1, B_1), new_size=size(A))
end

# ╔═╡ 44370313-62d3-4774-aed6-72a51410503d
img = Float64.(Gray.(testimage("house")));

# ╔═╡ d190772c-885f-405b-b0c9-bd11b6ac992c
img_blurry = Gray.(conv(kernel, img))

# ╔═╡ 0e57683e-7af8-4940-8494-032d6190f2cf
img_blurry[end-100:end-51, 1:50]

# ╔═╡ b8a8adfb-af4a-4ff9-b739-a040f05e2896
img_blurry2 = Gray.(conv_pad(img, kernel, 0.7, 10))

# ╔═╡ 94816b91-0ff3-41ef-9146-b71f150be161
img_blurry2[end-100:end-51, 1:50]

# ╔═╡ Cell order:
# ╠═0552a944-cdbe-11eb-2e63-49f12bee1624
# ╠═9a73fe7d-d298-4791-8f29-5a962d1242d1
# ╠═b8b2b314-4fed-420e-9061-6c2716c134c8
# ╠═97dc4b9d-0b32-4057-8a22-85cb40ae1ebf
# ╠═7500e8d2-16a7-4ea3-9192-78b4acfa327e
# ╠═44370313-62d3-4774-aed6-72a51410503d
# ╠═d190772c-885f-405b-b0c9-bd11b6ac992c
# ╠═0e57683e-7af8-4940-8494-032d6190f2cf
# ╠═b8a8adfb-af4a-4ff9-b739-a040f05e2896
# ╠═94816b91-0ff3-41ef-9146-b71f150be161
