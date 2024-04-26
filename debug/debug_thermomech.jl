include("../src/Mimosa.jl")
using Gridap
using TimerOutputs
using LinearAlgebra
using Test
using BenchmarkTools


 modelMR = Mimosa.MoneyRivlin3D(3.0, 1.0, 2.0)
 modelT = Mimosa.ThermalModel(1.0, 1.0, 2.0)

 function f(θ::Float64)
  return θ/1.0
 end
 df(θ::Float64)::Float64=1.0
modelTM = Mimosa.ThermoMech(modelT, modelMR, f, df)
 
Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ= modelTM(Mimosa.DerivativeStrategy{:analytic}())

θt = 3.4
∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
∇φ = VectorValue(1.0, 2.0, 3.0)
 
    (Ψ(∇u, θt))
   norm(∂Ψu(∇u, θt))
   norm(∂Ψθ(∇u, θt))
  norm(∂Ψuu(∇u, θt))
norm(∂Ψθθ(∇u, θt))
  norm(∂Ψuθ(∇u, θt))




 




 
