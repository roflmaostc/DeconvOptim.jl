using Documenter, DeconvOptim 


DocMeta.setdocmeta!(DeconvOptim, :DocTestSetup, :(using DeconvOptim); recursive=true)
makedocs(modules=[DeconvOptim],
         sitename="DeconvOptim.jl",
         doctest = false,
         pages = Any[
                "DeconvOptim.jl" => "index.md",
                "Basic Workflow" => "basic_workflow.md",
                "Background" => Any[
                    "background/physical_background.md",
                    "background/mathematical_optimization.md",
                    "background/regularizer.md",
                    ],
                "Function references" => Any[
                    "function_references/regularizer.md",
                    "function_references/utils.md",
                    ]
                ],
        )
         

deploydocs(repo = "github.com/roflmaostc/DeconvOptim.jl.git")
