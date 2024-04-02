include("../src/Mimosa.jl")
using Gridap
using TimerOutputs
using LinearAlgebra
using Test
using BenchmarkTools

modelNEO = Mimosa.NeoHookean3D(3.0, 1.0)
modelMR = Mimosa.MoneyRivlin3D(3.0, 1.0, 2.0)

Ψ_neo, ∂Ψu_neo, ∂Ψuu_neo = modelNEO(Mimosa.DerivativeStrategy{:analytic}())
Ψ_mr, ∂Ψu_mr, ∂Ψuu_mr= modelMR(Mimosa.DerivativeStrategy{:autodiff}())




# Ψ, ∂Ψu, ∂Ψuu= modelMR(Mimosa.DerivativeStrategy{:autodiff}())
∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3

res = @benchmark ∂Ψuu_mr(∇u)
@code_warntype ∂Ψuu_mr(∇u)

reset_timer!()
for i in 1:1e6
  ∂Ψuu_mr(∇u)
end
print_timer()

reset_timer!()
for i in 1:1e6
  ∂Ψuu_mr_(∇u)
end
print_timer()

