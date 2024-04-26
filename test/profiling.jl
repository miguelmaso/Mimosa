using Mimosa
using Profile
using Gridap

function main()

  
    # ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
    # F = one(∇u) + ∇u
    # J = det(F)
    #   J  
    #   logreg(J; Threshold=0.01) 
   
  
  # begin
  #   ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  #   modelLE = LinearElasticity3D(3.0, 1.0)
  #   Ψ, ∂Ψu, ∂Ψuu = modelLE(DerivativeStrategy{:analytic}())
  #   @show (Ψ(∇u)) 
  # end
  
  
  # begin
    ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
    model = NeoHookean3D(3.0, 1.0)
    Ψ, ∂Ψu, ∂Ψuu = model(DerivativeStrategy{:analytic}())
    return  ∂Ψuu(∇u)
  # end
  
#  begin
#     ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
#     model = MoneyRivlin3D(3.0, 1.0, 2.0)
#     Ψ, ∂Ψu, ∂Ψuu = model(DerivativeStrategy{:analytic}())
#     @show Ψ(∇u)  
#   end
  
  
#   begin
#     ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
#     ∇φ = VectorValue(1.0, 2.0, 3.0)
#     modelMR = MoneyRivlin3D(3.0, 1.0, 2.0)
#     modelID = IdealDielectric(4.0)
#     modelelectro = ElectroMech(modelMR, modelID)
#     Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelelectro(DerivativeStrategy{:analytic}())
#     @show Ψ(∇u, ∇φ)  
  
#   end
  
  
#   begin
#     ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
#     ∇φ = VectorValue(1.0, 2.0, 3.0)
#     θt = 3.4
#     modelMR = MoneyRivlin3D(3.0, 1.0, 2.0)
#     modelID = IdealDielectric(4.0)
#     modelT = ThermalModel(1.0, 1.0, 2.0)
#     f(θ::Float64)::Float64 = θ / 1.0
#     df(θ::Float64)::Float64 = 1.0
#     modelTEM = ThermoElectroMech(modelT, modelID, modelMR, f, df)
#     Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ = modelTEM(DerivativeStrategy{:analytic}())
#     @show (Ψ(∇u, ∇φ, θt)) 

#   end
  
  
  
  
#  begin
#     ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
#     ∇φ = VectorValue(1.0, 2.0, 3.0)
#     θt = 3.4
#     modelMR = MoneyRivlin3D(3.0, 1.0, 2.0)
#     modelT = ThermalModel(1.0, 1.0, 2.0)
#     f(θ::Float64)::Float64 = θ / 1.0
#     df(θ::Float64)::Float64 = 1.0
#     modelTM = ThermoMech(modelT, modelMR, f, df)
#     Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = modelTM(Mimosa.DerivativeStrategy{:analytic}())
  
#       @show Ψ(∇u, θt)
   
  
  
#   end
end


function I7()
b=get_array(TensorValue(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
end

function pp()
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  Ψ(∇u) = obj.μ / 2 * tr((F(∇u))' * F(∇u)) - obj.μ * log(J(F(∇u))) + (obj.λ / 2) * (J(F(∇u)) - 1)^2 - 3.0 * (obj.μ / 2.0)
∂Ψ_∂J(∇u) = -obj.μ / J(F(∇u)) + obj.λ * (J(F(∇u)) - 1)
∂Ψu(∇u) = obj.μ * F(∇u) + ∂Ψ_∂J(∇u) * H(F(∇u))
# I_ = TensorValue(Matrix(1.0I, 9, 9))
 ∂Ψ2_∂J2(∇u) = obj.μ / (J(F(∇u))^2) + obj.λ
# ∂Ψuu(∇u) =  I_#+ ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + ∂Ψ_∂J(∇u) * ×ᵢ⁴(F(∇u))
TensorValue(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
end
 
@time a=I7()

using PProf

PProf.Allocs.pprof(from_c=false)

using Gridap
  
using Profile
using StaticArrays

function I3()
  SMatrix{2,2}(1.0,1.0,1.0,1.0)
#  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0]
end
  using PProf

PProf.Allocs.pprof(from_c=false)

  @allocations a=I3()

  TensorValue(MMatrix{2,2}(1.0,1.0,1.0,1.0))
