using Pkg
using PackageCompiler
 

warmup_file = joinpath("/Users/jfrutos/Library/CloudStorage/Dropbox/julia_projects/Mimosa/examples/ex0/ex0p.jl")

pkgs = Symbol[]
push!(pkgs, :Mimosa)

if VERSION >= v"1.4"
    append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep],)
else
    append!(pkgs, [Symbol(name) for name in keys(Pkg.installed())])
end

create_sysimage(pkgs,
  sysimage_path=joinpath(@__DIR__,"Mimosa.so"),
  precompile_execution_file=warmup_file)