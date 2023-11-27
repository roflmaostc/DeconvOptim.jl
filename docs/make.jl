using Documenter, DocumenterCitations, DeconvOptim


cite_bib = CitationBibliography(joinpath(@__DIR__, "../paper/ref.bib"))

DocMeta.setdocmeta!(DeconvOptim, :DocTestSetup, :(using DeconvOptim); recursive=true)
makedocs(modules=[DeconvOptim],
         plugins=[cite_bib],
         sitename="DeconvOptim.jl",
         doctest = false,
         warnonly=true,
         pages = Any[
                "DeconvOptim.jl" => "index.md",
                "Workflow" => Any[
                        "workflow/basic_workflow.md",
                        "workflow/changing_regularizers.md",
                        "workflow/changing_loss.md",
                        "workflow/3D_dataset.md",
                        "workflow/cuda.md",
                        "workflow/flexible_invert.md",
                        "workflow/performance_tips.md",
                    ],
                "Background" => Any[
                    "background/physical_background.md",
                    "background/mathematical_optimization.md",
                    "background/loss_functions.md",
                    "background/regularizer.md",
                    ],
                "Function references" => Any[
                    "function_references/deconvolution.md",
                    "function_references/loss.md",
                    "function_references/mapping.md",
                    "function_references/regularizer.md",
                    "function_references/utils.md",
                    "function_references/analysis.md",
                    ],
                "References" => "references.md"
                ],
        )
         

deploydocs(repo = "github.com/roflmaostc/DeconvOptim.jl.git")
