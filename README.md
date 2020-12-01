# DeconvOptim.jl

<br>
<a name="logo"/>
<div align="left">
<a href="https://roflmaostc.github.io/DeconvOptim.jl/stable/" target="_blank">
<img src="docs/src/assets/logo.svg" alt="DeconvOptim Logo" width="150"></img>
</a>
</div>
<br>
A package for microscopy image based deconvolution via Optim.jl. This package works with N dimensional Point Spread Functions and images.
<br>

| **Documentation**                       | **Build Status**                          | **Code Coverage**               |
|:---------------------------------------:|:-----------------------------------------:|:-------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | ![CI][githubci-url] | [![][codecov-img]][codecov-url]|


## Documentation
The documentation of the latest release is [here](docs-stable-url).
The documentation of current master is [here](docs-dev-url).

## Installation
Type `]`in the REPL to get to the package manager:
```julia
julia> ] add DeconvOptim
```

## Usage
A quick example is shown below.
```julia
using Revise # for development useful
using DeconvOptim, TestImages, Images, FFTW, Noise, ImageView

# load test image
img = channelview(testimage("resolution_test_512"))

# generate simple Point Spread Function of aperture radius 30
psf = generate_psf(size(img), 30)

# create a blurred, noisy version of that image
img_b = conv_psf(img, psf)
img_n = poisson(img_b, 300)

# deconvolve 2D with default options
@time res, o = deconvolution(img_n, psf)

# show final results next to original and blurred version
imshow([img img_n res])
```
![Results Quick Example](docs/src/assets/quick_example_results.png)


## Development

The package is developed at [GitHub](https://www.github.com/roflmaostc/DeconvOptim.jl).  There
you can submit bug reports and make suggestions. 


## Contributions
I would like to thank Rainer Heintzmann for the great support and discussions during development.
Furthermore without [Tullio.jl](https://github.com/mcabbott/Tullio.jl) and [@mcabbott](https://github.com/mcabbott/) this package wouldn't be as fast as it is. His package and ideas are the basis for the implementations of the regularizers.


## To-Dos
* [ ] GPU support for improved version -> check Tullio for that. But won't be tackled soon.
* [ ] Enable Threaded Tullio again -> Currently disabled because of [this issue](https://github.com/mcabbott/Tullio.jl/issues/45)


[docs-dev-img]: https://img.shields.io/badge/docs-dev-orange.svg 
[docs-dev-url]: https://roflmaostc.github.io/DeconvOptim.jl/dev/ 

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg 
[docs-stable-url]: https://roflmaostc.github.io/DeconvOptim.jl/stable/

[travis-img]: https://travis-ci.com/roflmaostc/DeconvOptim.jl.svg?branch=master 
[travis-url]: https://travis-ci.com/github/roflmaostc/DeconvOptim.jl 

[githubci-url]: https://github.com/roflmaostc/DeconvOptim.jl/workflows/CI/badge.svg

[coveral-img]: https://coveralls.io/repos/github/roflmaostc/DeconvOptim.jl/badge.svg?branch=master
[coveral-url]: https://coveralls.io/github/roflmaostc/DeconvOptim.jl

[codecov-img]: https://codecov.io/gh/roflmaostc/DeconvOptim.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/roflmaostc/DeconvOptim.jl
