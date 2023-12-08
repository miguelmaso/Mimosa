using Mimosa
using Documenter

DocMeta.setdocmeta!(Mimosa, :DocTestSetup, :(using Mimosa); recursive=true)

makedocs(;
    modules=[Mimosa],
    authors="MultiSimo_Group",
    repo="https://github.com/jmartfrut/Mimosa.jl/blob/{commit}{path}#{line}",
    sitename="Mimosa.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jmartfrut.github.io/Mimosa.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jmartfrut/Mimosa.jl",
    devbranch="main",
)
