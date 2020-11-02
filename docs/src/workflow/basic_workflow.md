# Basic Workflow

In this section we show the workflow for deconvolution of 2D and 3D images using different regularizers. 
From these examples one can also see the different effects of the regularizers.

The picture below shows the general principle of DeconvOptim.jl.
The different parts of the pipeline can be exchanged and adapted to the users needs.
In most cases changing the regularizer or the iterations is enough.
However, exchanging all different parts is possible. For all options, see the function references.
Via the help of Julia (typing `]` in the REPL) we can also access extensive help.
