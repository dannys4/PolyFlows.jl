using PolyFlows
using Documenter

DocMeta.setdocmeta!(PolyFlows, :DocTestSetup, :(using PolyFlows); recursive=true)

makedocs(;
    modules=[PolyFlows],
    authors="Daniel Sharp <dannys4@mit.edu> and contributors",
    sitename="PolyFlows.jl",
    format=Documenter.HTML(;
        canonical="https://dannys4.github.io/PolyFlows.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dannys4/PolyFlows.jl",
    devbranch="main",
)
