### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 1d248a02-cdb3-11eb-25c2-552ed8a37503
begin
	import Pkg
    Pkg.activate(".")
end

# ╔═╡ 44be0bb5-155b-4311-aa9a-b0ad93ee5010
using Tullio, PGFPlotsX, OffsetArrays, FourierTools

# ╔═╡ 4c3c0f8a-9678-4584-9f6d-a2945a5250bb
begin
	conv_2(x,k) = @tullio y[i+_, j+_] := x[i+a, j+b] * k[a,b]
	conv_3(x,k) = @tullio y[i+_, j+_, l+_] := x[i+a, j+b, l+c] * k[a,b,c]
end

# ╔═╡ af2d1ca4-3835-4108-bd4c-9b080ce07c81
function extend_kernel(kernel, s)
	return collect(FourierTools.select_region(kernel, new_size=s))
end	

# ╔═╡ f54b482c-88d7-4e6f-8dd6-bdb01c17d15a
function compare_speed(N, K)
	img_2 = randn((N, N))
	kernel_2 = ones((K, K))
	kernel_full_2 = extend_kernel(kernel_2, (N, N))
	
	img_3 = randn((N, N, N))
	kernel_3 = ones((K, K, K))
	kernel_full_3 = extend_kernel(kernel_3, (N, N, N))
	
	v_ft, pconv = plan_conv(img_2, kernel_full_2) 
	@time res_conv = conv_2(img_2, kernel_2)
	@time res_fft = pconv(img_2)
	
	_, pconv3 = plan_conv(img_3, kernel_full_3) 
	@time res_conv = conv_3(img_3, kernel_3)
	@time res_fft = pconv3(img_3)
	
	end

# ╔═╡ d79ccda2-9905-46a4-8b91-d074e6fdb7d1
compare_speed(80, 5);

# ╔═╡ b6f7210e-6d8b-4919-af08-74239190afaf


# ╔═╡ ce20536f-5af4-42c8-b7d5-e89a4a473c67


# ╔═╡ b44602b9-074a-4507-bb04-c821b73bada9


# ╔═╡ Cell order:
# ╠═1d248a02-cdb3-11eb-25c2-552ed8a37503
# ╠═44be0bb5-155b-4311-aa9a-b0ad93ee5010
# ╠═4c3c0f8a-9678-4584-9f6d-a2945a5250bb
# ╠═af2d1ca4-3835-4108-bd4c-9b080ce07c81
# ╠═f54b482c-88d7-4e6f-8dd6-bdb01c17d15a
# ╠═d79ccda2-9905-46a4-8b91-d074e6fdb7d1
# ╠═b6f7210e-6d8b-4919-af08-74239190afaf
# ╠═ce20536f-5af4-42c8-b7d5-e89a4a473c67
# ╠═b44602b9-074a-4507-bb04-c821b73bada9
