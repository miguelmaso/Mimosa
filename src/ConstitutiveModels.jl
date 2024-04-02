module ConstitutiveModels

using Gridap
using ForwardDiff
using LinearAlgebra
using ..TensorAlgebra

export logreg
export NeoHookean3D
export MoneyRivlin3D
export DerivativeStrategy


abstract type ConstitutiveModel end
abstract type Mechanical <: ConstitutiveModel end
abstract type ElectroMechanical <: ConstitutiveModel end

struct DerivativeStrategy{Kind} end


# Jacobian regularization
function logreg(J; Threshold=0.01)
  if J >= Threshold
    return log(J)
  else
    return log(Threshold) - (3.0 / 2.0) + (2 / Threshold) * J - (1 / (2 * Threshold^2)) * J^2
  end
end




struct NeoHookean3D <: Mechanical
  λ::Float64
  μ::Float64
end

struct MoneyRivlin3D <: Mechanical
  λ::Float64
  μ1::Float64
  μ2::Float64
end

function (obj::NeoHookean3D)(strategy::DerivativeStrategy{:autodiff})
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  Ψ(∇u) = obj.μ / 2 * tr((F(∇u))' * F(∇u)) - obj.μ * logreg(J(F(∇u))) + (obj.λ / 2) * (J(F(∇u)) - 1)^2 - 3.0 * (obj.μ / 2.0)
  ∂Ψ_∂∇u(∇u) = ForwardDiff.gradient(∇u -> Ψ(∇u), get_array(∇u))
  ∂2Ψ_∂2∇u(∇u) = ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u), get_array(∇u))
  ∂Ψu(∇u) = TensorValue(∂Ψ_∂∇u(∇u))
  ∂Ψuu(∇u) = TensorValue(∂2Ψ_∂2∇u(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end

function (obj::NeoHookean3D)(strategy::DerivativeStrategy{:analytic})
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  Ψ(∇u) = obj.μ / 2 * tr((F(∇u))' * F(∇u)) - obj.μ * log(J(F(∇u))) + (obj.λ / 2) * (J(F(∇u)) - 1)^2 - 3.0 * (obj.μ / 2.0)
  ∂Ψ_∂J(∇u) = -obj.μ / J(F(∇u)) + obj.λ * (J(F(∇u)) - 1)
  ∂Ψu(∇u) = obj.μ * F(∇u) + ∂Ψ_∂J(∇u) * H(F(∇u))
  I_ = TensorValue(Matrix(1.0I, 9, 9))
  ∂Ψ2_∂J2(∇u) = obj.μ / (J(F(∇u))^2) + obj.λ
  ∂Ψuu(∇u) = obj.μ * I_ + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + ∂Ψ_∂J(∇u) * cross_I4(F(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end


function (obj::MoneyRivlin3D)(strategy::DerivativeStrategy{:autodiff})
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  Ψ(∇u) = obj.μ1 / 2 * tr((F(∇u))' * F(∇u)) + obj.μ2 / 2 * tr((H(F(∇u)))' * H(F(∇u))) - (obj.μ1 + 2 * obj.μ2) * logreg(J(F(∇u))) +
          (obj.λ / 2) * (J(F(∇u)) - 1)^2 - (3.0 / 2.0) * (obj.μ1 + obj.μ2)
  ∂Ψ_∂∇u(∇u) = ForwardDiff.gradient(∇u -> Ψ(∇u), get_array(∇u))
  ∂2Ψ_∂2∇u(∇u) = TensorValue(ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u), get_array(∇u)))
  ∂Ψu(∇u) = TensorValue(∂Ψ_∂∇u(∇u))
  ∂Ψuu(∇u) = TensorValue(∂2Ψ_∂2∇u(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end


function (obj::MoneyRivlin3D)(strategy::DerivativeStrategy{:analytic})
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  Ψ(∇u) = obj.μ1 / 2 * tr((F(∇u))' * F(∇u)) + obj.μ2 / 2.0 * tr((H(F(∇u)))' * H(F(∇u))) - (obj.μ1 + 2 * obj.μ2) * log(J(F(∇u))) +
          (obj.λ / 2.0) * (J(F(∇u)) - 1)^2 - (3.0 / 2.0) * (obj.μ1 + obj.μ2)
  ∂Ψ_∂F(∇u) = obj.μ1 * F(∇u)
  ∂Ψ_∂H(∇u) = obj.μ2 * H(F(∇u))
  ∂Ψ_∂J(∇u) = -(obj.μ1 + 2.0 * obj.μ2) / J(F(∇u)) + obj.λ * (J(F(∇u)) - 1)
  ∂Ψ2_∂J2(∇u) = (obj.μ1 + 2.0 * obj.μ2) / (J(F(∇u))^2) + obj.λ
  ∂Ψu(∇u) = ∂Ψ_∂F(∇u) + ∂Ψ_∂H(∇u) × F(∇u) + ∂Ψ_∂J(∇u) * H(F(∇u))
  I_ = TensorValue(Matrix(1.0I, 9, 9))
  # ∂Ψuu(∇u) = obj.μ1 * I_ + obj.μ2 * (F(∇u) × (I_ × F(∇u))) + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + (I_ × (∂Ψ_∂H(∇u) + ∂Ψ_∂J(∇u) * F(∇u)))
  ∂Ψuu(∇u) = obj.μ1 * I_ + obj.μ2 * (F(∇u) × (I_ × F(∇u))) + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + cross_I4(∂Ψ_∂H(∇u) + ∂Ψ_∂J(∇u) * F(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end







end