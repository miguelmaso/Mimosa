module ConstitutiveModels

using Gridap
using ForwardDiff
using LinearAlgebra
using StaticArrays
using ..TensorAlgebra
using ..TensorAlgebra: _δδ_μ_3D
using ..TensorAlgebra: _δδ_λ_3D
using ..TensorAlgebra: _δδ_μ_2D
using ..TensorAlgebra: _δδ_λ_2D
using ..TensorAlgebra: I3
using ..TensorAlgebra: I9

export NeoHookean3D
export MoneyRivlin3D
export LinearElasticity3D
export IdealDielectric
export ThermalModel
export ElectroMech
export ThermoElectroMech
export ThermoMech
export Mechano
export Elasto
export Visco
export ViscoElastic
export GeneralizedMaxwell
export ViscousIncompressible
export IsochoricNeoHookean3D
export initializeStateVariables
export setHistoricalVariables
export updateStateVariables!
export DerivativeStrategy
export StressTensor

struct DerivativeStrategy{Kind} end
struct StressTensor{Kind} end

abstract type ConstitutiveModel end
abstract type Mechano <: ConstitutiveModel end
abstract type Electro <: ConstitutiveModel end
abstract type Magneto <: ConstitutiveModel end
abstract type Thermo <: ConstitutiveModel end
abstract type Multiphysic <: ConstitutiveModel end
abstract type Elasto <: Mechano end
abstract type Visco <: Mechano end
abstract type ViscoElastic <: Mechano end


# ===================
# Electro models
# ===================

@kwdef struct IdealDielectric <: Electro
  ε::Float64
end

# ===================
# Thermal models
# ===================

@kwdef struct ThermalModel <: Thermo
  Cv::Float64
  θr::Float64
  α::Float64
  κ::Float64=10.0
end

# ===================
# Mechanical models
# ===================

@kwdef struct LinearElasticity3D <: Elasto
  λ::Float64
  μ::Float64
  ρ::Float64=0.0
end

@kwdef struct NeoHookean3D <: Elasto
  λ::Float64
  μ::Float64
  ρ::Float64=0.0
end

@kwdef struct MoneyRivlin3D <: Elasto
  λ::Float64
  μ1::Float64
  μ2::Float64
  ρ::Float64=0.0
end

@kwdef struct IsochoricNeoHookean3D <: Elasto
  μ::Float64
end

# ===================
# Visco elastic models
# ===================

@kwdef struct ViscousIncompressible <: Visco
  ShortTerm::Elasto
  τ::Float64
end

@kwdef struct GeneralizedMaxwell <: ViscoElastic
  LongTerm::Elasto
  Branch1::Visco
  Branches::Array{Visco}
  function GeneralizedMaxwell(LongTerm::Elasto,Branches::Visco...)
    new(LongTerm,first(Branches),collect(Branches))
  end
end

# ===================
# Multiphysic models
# ===================

struct ElectroMech <: Multiphysic
  Model1::Mechano
  Model2::Electro
end

struct ThermoMech <: Multiphysic
  Model1::Thermo
  Model2::Mechano
  fθ::Function
  dfdθ::Function
end

struct ThermoElectroMech <: Multiphysic
  Model1::Thermo
  Model2::Electro
  Model3::Mechano
  fθ::Function
  dfdθ::Function
end


function _getKinematic(::Mechano)
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  return (F, H, J)
end

function _getKinematic(::Electro)
  E(∇φ) = -∇φ
  return E
end

function _getKinematic(::Visco)
  F(∇u) = one(∇u) + ∇u
  C(F) = F' * F
  Ce(C,Uvα⁻¹) = Uvα⁻¹ * C * Uvα⁻¹
  return (F, C, Ce)
end


# ===============================
# Coupling terms for multiphysic
# ===============================

function _getCoupling(mech::Mechano, elec::Electro)
  F, H, J = _getKinematic(mech)
  E = _getKinematic(elec)

  # Energy #
  HE(∇u, ∇φ) = H(F(∇u)) * E(∇φ)
  HEHE(∇u, ∇φ) = HE(∇u, ∇φ) ⋅ HE(∇u, ∇φ)
  Ψem(∇u, ∇φ) = (-elec.ε / (2.0 * J(F(∇u)))) * HEHE(∇u, ∇φ)
  # First Derivatives #
  ∂Ψem_∂H(∇u, ∇φ) = (-elec.ε / (J(F(∇u)))) * (HE(∇u, ∇φ) ⊗ E(∇φ))
  ∂Ψem_∂J(∇u, ∇φ) = (+elec.ε / (2.0 * J(F(∇u))^2.0)) * HEHE(∇u, ∇φ)
  ∂Ψem_∂E(∇u, ∇φ) = (-elec.ε / (J(F(∇u)))) * (H(F(∇u))' * HE(∇u, ∇φ))
  ∂Ψem_u(∇u, ∇φ) = ∂Ψem_∂H(∇u, ∇φ) × F(∇u) + ∂Ψem_∂J(∇u, ∇φ) * H(F(∇u))
  ∂Ψem_φ(∇u, ∇φ) = -∂Ψem_∂E(∇u, ∇φ)
  # Second Derivatives #
  # I33 = TensorValue(Matrix(1.0I, 3, 3))
  I33=I3()
  ∂Ψem_HH(∇u, ∇φ) = (-elec.ε / (J(F(∇u)))) * (I33 ⊗₁₃²⁴ (E(∇φ) ⊗ E(∇φ)))
  ∂Ψem_HJ(∇u, ∇φ) = (+elec.ε / (J(F(∇u)))^2.0) * (HE(∇u, ∇φ) ⊗ E(∇φ))
  ∂Ψem_JJ(∇u, ∇φ) = (-elec.ε / (J(F(∇u)))^3.0) * HEHE(∇u, ∇φ)
  ∂Ψem_uu(∇u, ∇φ) = (F(∇u) × (∂Ψem_HH(∇u, ∇φ) × F(∇u))) +
                    H(F(∇u)) ⊗₁₂³⁴ (∂Ψem_HJ(∇u, ∇φ) × F(∇u)) +
                    (∂Ψem_HJ(∇u, ∇φ) × F(∇u)) ⊗₁₂³⁴ H(F(∇u)) +
                    ∂Ψem_JJ(∇u, ∇φ) * (H(F(∇u)) ⊗₁₂³⁴ H(F(∇u))) +
                    ×ᵢ⁴(∂Ψem_∂H(∇u, ∇φ) + ∂Ψem_∂J(∇u, ∇φ) * F(∇u))

  ∂Ψem_EH(∇u, ∇φ) = (-elec.ε / (J(F(∇u)))) * ((I33 ⊗₁₃² HE(∇u, ∇φ)) + (H(F(∇u))' ⊗₁₂³ E(∇φ)))
  ∂Ψem_EJ(∇u, ∇φ) = (+elec.ε / (J(F(∇u)))^2.0) * (H(F(∇u))' * HE(∇u, ∇φ))

  ∂Ψem_φu(∇u, ∇φ) = -(∂Ψem_EH(∇u, ∇φ) × F(∇u)) - (∂Ψem_EJ(∇u, ∇φ) ⊗₁²³ H(F(∇u)))
  ∂Ψem_φφ(∇u, ∇φ) = (-elec.ε / (J(F(∇u)))) * (H(F(∇u))' * H(F(∇u)))

  return (Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ)

end

function _getCoupling(mech::Mechano, term::Thermo)
  F, H, J = _getKinematic(mech)
  
  ∂Ψtm_∂J(∇u, δθ) = -6.0 * term.α * J(F(∇u)) * δθ
  ∂Ψtm_u(∇u, δθ) = ∂Ψtm_∂J(∇u, δθ) * H(F(∇u))
  ∂Ψtm_θ(∇u, δθ) = -3.0 * term.α * (J(F(∇u))^2.0 - 1.0)
  ∂Ψtm_uu(∇u, δθ) = (-6.0 * term.α * δθ) * (H(F(∇u)) ⊗₁₂³⁴ H(F(∇u))) + ×ᵢ⁴(∂Ψtm_∂J(∇u, δθ) * F(∇u))
  ∂Ψtm_uθ(∇u, δθ) = -6.0 * term.α * J(F(∇u)) * H(F(∇u))
  ∂Ψtm_θθ(∇u, δθ) = 0.0

  Ψtm(∇u, δθ) = ∂Ψtm_θ(∇u, δθ) * δθ

  return (Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ)
end


# ====================
# Constitutive models
# ====================

include("./ViscousModels.jl")

function (obj::ViscousIncompressible)(strategy::DerivativeStrategy{T}) where T
  Ψe, Se, ∂Se∂Ce          = obj.ShortTerm(strategy, StressTensor{:SecondPiola}())
  ∂Ψ∂F(∇u, ∇un, Δt, state)   = Piola(obj, Se, ∂Se∂Ce, ∇u, ∇un, Δt, state)
  ∂Ψ∂F∂F(∇u, ∇un, Δt, state) = Tangent(obj, Se, ∂Se∂Ce, ∇u, ∇un, Δt, state)
  return Ψe, ∂Ψ∂F, ∂Ψ∂F∂F
end


function (obj::GeneralizedMaxwell)(strategy::DerivativeStrategy{T}) where T
  Ψe, ∂Ψeu, ∂Ψeuu = obj.LongTerm(strategy)
  # Ψ1, ∂Ψ1u, ∂Ψ1uu = obj.Branch1(strategy)
  # Ψ(∇u, ∇un, Δt, state) = Ψe(∇u) + Ψ1(∇u, ∇un, Δt, state)
  # ∂Ψu(∇u, ∇un, Δt, state) = ∂Ψeu(∇u) + ∂Ψ1u(∇u, ∇un, Δt, state)
  # ∂Ψuu(∇u, ∇un, Δt, state) = ∂Ψeuu(∇u) + ∂Ψ1uu(∇u, ∇un, Δt, state)
  # return (Ψ, ∂Ψu, ∂Ψuu)
  N = length(obj.Branches)
  Ψα = Array{Function}(undef,N)
  ∂Ψαu = Array{Function}(undef,N)
  ∂Ψαuu = Array{Function}(undef,N)
  for (i, branch) in enumerate(obj.Branches)
    Ψi, ∂Ψiu, ∂Ψiuu = branch(strategy)
    Ψα[i] = Ψi
    ∂Ψαu[i] = ∂Ψiu
    ∂Ψαuu[i] = ∂Ψiuu
  end
  Ψ(∇u, ∇un, Δt, states...) = mapreduce((Ψi, state) -> Ψi(∇u, ∇un, Δt, state), +, Ψα, states; init=Ψe(∇u))
  ∂Ψu(∇u, ∇un, Δt, states...) = mapreduce((∂Ψiu, state) -> ∂Ψiu(∇u, ∇un, Δt, state), +, ∂Ψαu, states; init=∂Ψeu(∇u))
  ∂Ψuu(∇u, ∇un, Δt, states...) = mapreduce((∂Ψiuu, state) -> ∂Ψiuu(∇u, ∇un, Δt, state), +, ∂Ψαuu, states; init=∂Ψeuu(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end


function (obj::LinearElasticity3D)(::DerivativeStrategy{:analytic})
  F, _, _ = _getKinematic(obj)
  # I33 = TensorValue(Matrix(1.0I, 3, 3))
  I33=I3()
  ∂Ψuu(∇u) = _δδ_μ_3D(obj.μ) + _δδ_λ_3D(obj.λ)
  ∂Ψu(∇u) = ∂Ψuu(∇u) ⊙ (F(∇u) - I33)
  Ψ(∇u) = 0.5 * (F(∇u) - I33) ⊙ (∂Ψuu(∇u) ⊙ (F(∇u) - I33))
  return (Ψ, ∂Ψu, ∂Ψuu)
end

function (obj::NeoHookean3D)(::DerivativeStrategy{:autodiff})
  F, _, J = _getKinematic(obj)
  Ψ(∇u) = obj.μ / 2 * tr((F(∇u))' * F(∇u)) - obj.μ * logreg(J(F(∇u))) + (obj.λ / 2) * (J(F(∇u)) - 1)^2 - 3.0 * (obj.μ / 2.0)
  ∂Ψ_∂∇u(∇u) = ForwardDiff.gradient(∇u -> Ψ(∇u), get_array(∇u))
  ∂2Ψ_∂2∇u(∇u) = ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u), get_array(∇u))
  ∂Ψu(∇u) = TensorValue(∂Ψ_∂∇u(∇u))
  ∂Ψuu(∇u) = TensorValue(∂2Ψ_∂2∇u(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end

function (obj::NeoHookean3D)(::DerivativeStrategy{:analytic})
  F, H, J = _getKinematic(obj)
  Ψ(∇u) = obj.μ / 2 * tr((F(∇u))' * F(∇u)) - obj.μ * log(J(F(∇u))) + (obj.λ / 2) * (J(F(∇u)) - 1)^2 - 3.0 * (obj.μ / 2.0)
  ∂Ψ_∂J(∇u) = -obj.μ / J(F(∇u)) + obj.λ * (J(F(∇u)) - 1)
  ∂Ψu(∇u) = obj.μ * F(∇u) + ∂Ψ_∂J(∇u) * H(F(∇u))
  # I_ = TensorValue(Matrix(1.0I, 9, 9))
   I_=I9()
  ∂Ψ2_∂J2(∇u) = obj.μ / (J(F(∇u))^2) + obj.λ
  ∂Ψuu(∇u) = obj.μ * I_ + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + ∂Ψ_∂J(∇u) * ×ᵢ⁴(F(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end


function (obj::IsochoricNeoHookean3D)(strategy::DerivativeStrategy{T}) where T
  obj(strategy, StressTensor{FirstPiola}())
end


function (obj::IsochoricNeoHookean3D)(::DerivativeStrategy{T}, ::StressTensor{:FirstPiola}) where T
  F, H, J = _getKinematic(obj)
  I = I9()
  Ψ(∇u)       =  obj.μ / 2 * tr((F(∇u))' * F(∇u)) - obj.μ * log(J(F(∇u))) - 3.0 * (obj.μ / 2.0)
  ∂Ψ_∂J(∇u)   = -obj.μ / J(F(∇u))
  ∂Ψu(∇u)     =  obj.μ * F(∇u) + ∂Ψ_∂J(∇u) * H(F(∇u))
  ∂Ψ2_∂J2(∇u) =  obj.μ / (J(F(∇u))^2)
  ∂Ψuu(∇u)    =  obj.μ * I + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + ∂Ψ_∂J(∇u) * ×ᵢ⁴(F(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end


function (obj::IsochoricNeoHookean3D)(::DerivativeStrategy{T}, ::StressTensor{:SecondPiola}) where T
  Ψe(Ce)     = obj.μ / 2 * tr(Ce) * det(Ce)^(-1/3)
  Se_(Ce)    = 2 * ForwardDiff.gradient(Ce -> Ψe(Ce), get_array(Ce))
  ∂Se∂Ce_(Ce) = ForwardDiff.jacobian(Ce -> Se_(Ce), get_array(Ce))
  Se = Se_
  ∂Se∂Ce = ∂Se∂Ce_
  # Se(Ce)     = TensorValue(Se_(Ce))
  # ∂Se∂Ce(Ce) = TensorValue(∂Se∂Ce_(Ce))
  # TODO: Check correctness of analytical derivatives
  # Se(Ce)     = obj.μ * det(Ce)^(-1/3) * I - obj.μ / 3 * tr(Ce) * det(Ce)^(-1/3) * inv(Ce')
  # ∂Se∂Ce(Ce) = -2/3 * obj.μ * det(Ce)^(-1/3) * inv(Ce') ⊗ I -2/9 * obj.μ * tr(Ce) * det(Ce)^(-1/3) * inv(Ce') ⊗ inv(Ce')
  # ∂Se∂Ce(Ce) = inv(Ce') |> invCe -> -2/3 * obj.μ * det(Ce)^(-1/3) * invCe ⊗ I -2/9 * obj.μ * tr(Ce) * det(Ce)^(-1/3) * invCe ⊗ invCe
  # ∂Se∂Ce(Ce) = inv(Ce') |> invCe -> -2/3 * obj.μ * det(Ce)^(-1/3) * invCe ⊗ (I + 1/3 * tr(Ce) * invCe)
  return (Ψe, Se, ∂Se∂Ce)
end


function (obj::MoneyRivlin3D)(::DerivativeStrategy{:autodiff})
  F, H, J = _getKinematic(obj)
  Ψ(∇u) = obj.μ1 / 2 * tr((F(∇u))' * F(∇u)) + obj.μ2 / 2 * tr((H(F(∇u)))' * H(F(∇u))) - (obj.μ1 + 2 * obj.μ2) * logreg(J(F(∇u))) +
          (obj.λ / 2) * (J(F(∇u)) - 1)^2 - (3.0 / 2.0) * (obj.μ1 + obj.μ2)
  ∂Ψ_∂∇u(∇u) = ForwardDiff.gradient(∇u -> Ψ(∇u), get_array(∇u))
  ∂2Ψ_∂2∇u(∇u) = TensorValue(ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u), get_array(∇u)))
  ∂Ψu(∇u) = TensorValue(∂Ψ_∂∇u(∇u))
  ∂Ψuu(∇u) = TensorValue(∂2Ψ_∂2∇u(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end

function (obj::MoneyRivlin3D)(::DerivativeStrategy{:analytic})
  F, H, J = _getKinematic(obj)
  Ψ(∇u) = obj.μ1 / 2 * tr((F(∇u))' * F(∇u)) + obj.μ2 / 2.0 * tr((H(F(∇u)))' * H(F(∇u))) - (obj.μ1 + 2 * obj.μ2) * log(J(F(∇u))) +
          (obj.λ / 2.0) * (J(F(∇u)) - 1)^2 - (3.0 / 2.0) * (obj.μ1 + obj.μ2)
  ∂Ψ_∂F(∇u) = obj.μ1 * F(∇u)
  ∂Ψ_∂H(∇u) = obj.μ2 * H(F(∇u))
  ∂Ψ_∂J(∇u) = -(obj.μ1 + 2.0 * obj.μ2) / J(F(∇u)) + obj.λ * (J(F(∇u)) - 1)
  ∂Ψ2_∂J2(∇u) = (obj.μ1 + 2.0 * obj.μ2) / (J(F(∇u))^2) + obj.λ
  ∂Ψu(∇u) = ∂Ψ_∂F(∇u) + ∂Ψ_∂H(∇u) × F(∇u) + ∂Ψ_∂J(∇u) * H(F(∇u))
  # I_ = TensorValue(Matrix(1.0I, 9, 9))
  I_=I9()
  # ∂Ψuu(∇u) = obj.μ1 * I_ + obj.μ2 * (F(∇u) × (I_ × F(∇u))) + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + (I_ × (∂Ψ_∂H(∇u) + ∂Ψ_∂J(∇u) * F(∇u)))
  ∂Ψuu(∇u) = obj.μ1 * I_ + obj.μ2 * (F(∇u) × (I_ × F(∇u))) + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + ×ᵢ⁴(∂Ψ_∂H(∇u) + ∂Ψ_∂J(∇u) * F(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end

function (obj::ElectroMech)(strategy::DerivativeStrategy{:analytic})
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Model1(strategy)
  Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.Model1, obj.Model2)

  Ψ(∇u, ∇φ) = Ψm(∇u) + Ψem(∇u, ∇φ)
  ∂Ψu(∇u, ∇φ) = ∂Ψm_u(∇u) + ∂Ψem_u(∇u, ∇φ)
  ∂Ψφ(∇u, ∇φ) = ∂Ψem_φ(∇u, ∇φ)
  ∂Ψuu(∇u, ∇φ) = ∂Ψm_uu(∇u) + ∂Ψem_uu(∇u, ∇φ)
  ∂Ψφu(∇u, ∇φ) = ∂Ψem_φu(∇u, ∇φ)
  ∂Ψφφ(∇u, ∇φ) = ∂Ψem_φφ(∇u, ∇φ)

  return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
end

function (obj::ThermoMech)(strategy::DerivativeStrategy{:analytic})
  Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.Model1(strategy)
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Model2(strategy)
  Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.Model2, obj.Model1)

  f(δθ) = (obj.fθ(δθ)::Float64)
  df(δθ) = (obj.dfdθ(δθ)::Float64)

  Ψ(∇u, δθ) = f(δθ) * (Ψm(∇u)) + (Ψt(δθ) + Ψtm(∇u, δθ))
  ∂Ψu(∇u, δθ) = f(δθ) * (∂Ψm_u(∇u)) + ∂Ψtm_u(∇u, δθ)
  ∂Ψθ(∇u, δθ) = df(δθ) * (Ψm(∇u)) + ∂Ψtm_θ(∇u, δθ) + ∂Ψt_θ(δθ)

  ∂Ψuu(∇u, δθ) = f(δθ) * (∂Ψm_uu(∇u)) + ∂Ψtm_uu(∇u, δθ)
  ∂Ψθθ(∇u, δθ) = ∂Ψtm_θθ(∇u, δθ) + ∂Ψt_θθ(δθ)
  ∂Ψuθ(∇u, δθ) = df(δθ) * (∂Ψm_u(∇u)) + ∂Ψtm_uθ(∇u, δθ)

  return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ)
end

function (obj::ThermoElectroMech)(strategy::DerivativeStrategy{:analytic})
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Model3(strategy)
  Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.Model1(strategy)
  Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.Model3, obj.Model2)
  Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.Model3, obj.Model1)
  f(δθ) = (obj.fθ(δθ)::Float64)
  df(δθ) = (obj.dfdθ(δθ)::Float64)

  Ψ(∇u, ∇φ, δθ) = f(δθ) * (Ψm(∇u) + Ψem(∇u, ∇φ)) + (Ψt(δθ) + Ψtm(∇u, δθ))
  ∂Ψu(∇u, ∇φ, δθ) = f(δθ) * (∂Ψm_u(∇u) + ∂Ψem_u(∇u, ∇φ)) + ∂Ψtm_u(∇u, δθ)
  ∂Ψφ(∇u, ∇φ, δθ) = f(δθ) * ∂Ψem_φ(∇u, ∇φ)
  ∂Ψθ(∇u, ∇φ, δθ) = df(δθ) * (Ψm(∇u) + Ψem(∇u, ∇φ)) + ∂Ψtm_θ(∇u, δθ) + ∂Ψt_θ(δθ)


  ∂Ψuu(∇u, ∇φ, δθ) = f(δθ) * (∂Ψm_uu(∇u) + ∂Ψem_uu(∇u, ∇φ)) + ∂Ψtm_uu(∇u, δθ)
  ∂Ψφu(∇u, ∇φ, δθ) = f(δθ) * ∂Ψem_φu(∇u, ∇φ)
  ∂Ψφφ(∇u, ∇φ, δθ) = f(δθ) * ∂Ψem_φφ(∇u, ∇φ)
  ∂Ψθθ(∇u, ∇φ, δθ) = ∂Ψtm_θθ(∇u, δθ) + ∂Ψt_θθ(δθ)
  ∂Ψuθ(∇u, ∇φ, δθ) = df(δθ) * (∂Ψm_u(∇u) + ∂Ψem_u(∇u, ∇φ)) + ∂Ψtm_uθ(∇u, δθ)
  ∂Ψφθ(∇u, ∇φ, δθ) = df(δθ) * ∂Ψem_φ(∇u, ∇φ)

  return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ)
end

function (obj::ThermalModel)(::DerivativeStrategy{:analytic})
  Ψ(δθ) = obj.Cv * (δθ - (δθ+obj.θr) * log((δθ+obj.θr) / obj.θr))
  ∂Ψθ(δθ) = -obj.Cv * log((δθ+obj.θr) / obj.θr)
  ∂Ψθθ(δθ) = -obj.Cv / (δθ+obj.θr)
  return (Ψ, ∂Ψθ, ∂Ψθθ)
end




# Jacobian regularization
function logreg(J; Threshold=0.01)
  if J >= Threshold
    return log(J)
  else
    return log(Threshold) - (3.0 / 2.0) + (2 / Threshold) * J - (1 / (2 * Threshold^2)) * J^2
  end
end


# ======================================
# Material formulations: State variables
# ======================================

# --------------------------------------
# State variables initialization
# --------------------------------------

function initializeStateVariables(::ConstitutiveModel, points::Measure)
  return nothing
end

function initializeStateVariables(model::GeneralizedMaxwell, points::Measure)
  # initializeStateVariables(model.Branch1, points)
  states = []
  for branch in model.Branches
    push!(states, initializeStateVariables(branch, points))
  end
  return states
  # map(b -> initializeStateVariables(b, points), model.Branches)  #TODO: Check alternative syntax
end

function initializeStateVariables(::ViscousIncompressible, points::Measure)
  # v = SMatrix{3,3}(I)
  # Uv = CellState(v, points)
  # λv = CellState(0, points)
  # return (; Uv, λv)
  # TODO: Check if named tuple can be used instead of a vector
  v = VectorValue(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0)
  CellState(v, points)
end


# --------------------------------------
# State variables update
# --------------------------------------

function updateStateVariables!(::ConstitutiveModel, vars...)
end

function updateStateVariables!(model::GeneralizedMaxwell, u, un, Δt, stateVars)
  # updateStateVariables!(model.Branch1, ∇u, ∇un, Δt, stateVars...)
  @assert length(model.Branches) == length(stateVars)
  for (branch, state) in zip(model.Branches, stateVars)
    updateStateVariables!(branch, u, un, Δt, state)
  end
end

function updateStateVariables!(model::ViscousIncompressible, u, un, Δt, stateVar)
  _, Se, ∂Se∂Ce = model.ShortTerm(DerivativeStrategy{:analytic}(), StressTensor{:SecondPiola}())
  return_mapping(s, ∇u, ∇un) = ReturnMapping(model, Se, ∂Se∂Ce, ∇u, ∇un, Δt, s)
  update_state!(return_mapping, stateVar, ∇(u)', ∇(un)')
end


end