# Basic Workflow

In this section we show the workflow for deconvolution of 2D and 3D images using different regularizers. 
From these examples one can also see the different effects of the regularizers.

The picture below shows the general principle of DeconvOptim.jl.
Since we interprete deconvolution as optimization we initialize the reconstruction variables *rec*. We do have as many variables as pixels in the measured data.
Then we apply some mapping eg to reconstruct only pixels having non-negative intensity value.
After we compose the loss functions. It consists of a regularizer part (weighted with $\lambda$) and a loss part.
The latter one compares the current reconstruction with the measured image.
Total loss combines both values to a single scalar value. Using Zygote.jl to calculate the gradient with respect to all pixel values of rec, we can
plug the gradient and the loss function into Optim.jl. Optim.jl then minimizes this loss function.
The different parts of the pipeline (mapping, forward, regularizer) can be exchanged and adapted to the users needs.
In most cases changing the regularizer or the number of iterations is enough.


![](../assets/tex/pipeline.svg)

For all options, see the function references.
Via the help of Julia (typing `]` in the REPL) we can also access extensive help.
