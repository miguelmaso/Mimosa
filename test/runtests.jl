using Mimosa
using Test
using Profile

@time begin
    include("ConstitutiveModels.jl")
end
@time begin
    include("TensorAlgebra.jl")
end
@time begin
    include("WeakForms.jl")
end
