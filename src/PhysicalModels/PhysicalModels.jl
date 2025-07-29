module PhysicalModels

using Gridap
using ForwardDiff
using LinearAlgebra
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
export ElectroMechModel
export ThermoElectroMechModel
export ThermoMechModel
export ThermoMech_EntropicPolyconvex

export Mechano
export Electro
export Magneto
export Thermo
export ElectroMechano
export ThermoElectroMechano
export ThermoMechano
export ThermoElectro


export DerivativeStrategy

struct DerivativeStrategy{Kind} end

abstract type PhysicalModel end
abstract type Mechano <: PhysicalModel end
abstract type Electro <: PhysicalModel end
abstract type Magneto <: PhysicalModel end
abstract type Thermo <: PhysicalModel end

abstract type MultiPhysicalModel <: PhysicalModel end
abstract type ElectroMechano <: MultiPhysicalModel end
abstract type ThermoElectroMechano <: MultiPhysicalModel end
abstract type ThermoMechano <: MultiPhysicalModel end
abstract type ThermoElectro <: MultiPhysicalModel end



# ===================
# Electro models
# ===================

 struct IdealDielectric <: Electro
  ε::Float64
  function IdealDielectric(; ε::Float64)  
    new(ε)
  end
end

# ===================
# Thermal models
# ===================

struct ThermalModel <: Thermo
  Cv::Float64
  θr::Float64
  α::Float64
  κ::Float64
  function ThermalModel(; Cv::Float64, θr::Float64, α::Float64, κ::Float64=10.0)  
    new(Cv, θr, α, κ)
  end
end


# ===================
# Mechanical models
# ===================

struct LinearElasticity3D <: Mechano
  λ::Float64
  μ::Float64
  ρ::Float64
  function LinearElasticity3D(; λ::Float64, μ::Float64, ρ::Float64=0.0)  
    new(λ, μ, ρ)
  end
end

struct NeoHookean3D <: Mechano
  λ::Float64
  μ::Float64
  ρ::Float64
  function NeoHookean3D(; λ::Float64, μ::Float64, ρ::Float64=0.0)  
    new(λ, μ, ρ)
  end
end

struct MoneyRivlin3D <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  ρ::Float64
  function MoneyRivlin3D(; λ::Float64, μ1::Float64, μ2::Float64, ρ::Float64=0.0)  
    new(λ, μ1, μ2, ρ)
  end
end

# ===================
# MultiPhysicalModel models
# ===================

struct ElectroMechModel <: ElectroMechano
  Mechano::Mechano
  Electro::Electro
  function ElectroMechModel(; Mechano::Mechano, Electro::Electro)  
    new(Mechano, Electro)
  end
end

struct ThermoMechModel <: ThermoMechano
  Thermo::Thermo
  Mechano::Mechano
  fθ::Function
  dfdθ::Function
  function ThermoMechModel(; Thermo::Thermo, Mechano::Mechano, fθ::Function, dfdθ::Function)  
    new(Thermo, Mechano, fθ, dfdθ)
  end


end

struct ThermoMech_EntropicPolyconvex <: ThermoMechano
  Thermo::Thermo
  Mechano::Mechano
  β::Float64
  G::Function
  ϕ::Function
  s::Function
  function ThermoMech_EntropicPolyconvex(; Thermo::Thermo, Mechano::Mechano, β::Float64, G::Function, ϕ::Function, s::Function)  
    new(Thermo, Mechano, β, G, ϕ, s)
  end

end

struct ThermoElectroMechModel <: ThermoElectroMechano
  Thermo::Thermo
  Electro::Electro
  Mechano::Mechano
  fθ::Function
  dfdθ::Function
  function ThermoElectroMechModel(; Thermo::Thermo, Electro::Electro, Mechano::Mechano, fθ::Function, dfdθ::Function)  
    new(Thermo, Electro, Mechano, fθ, dfdθ)
  end
end

function _getKinematic(::Mechano)
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  return (F, H, J)
end

function _getInvariants_Isotropic(model::Mechano)
  F, H, J = _getKinematic(model)
  I1(∇u) = tr(F(∇u)' * F(∇u))
  I2(∇u) = tr(H(F(∇u))' * H(F(∇u)))
  I3(∇u) = J(F(∇u))
  return (I1, I2, I3)
end

function _getKinematic(::Electro)
  E(∇φ) = -∇φ
  return E
end


# ===============================
# Coupling terms for multiphysic
# ===============================

function _getCoupling(mech::Mechano, elec::IdealDielectric)
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
  I33 = I3()
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

function (obj::LinearElasticity3D)(::DerivativeStrategy{:analytic})
  F, _, _ = _getKinematic(obj)
  # I33 = TensorValue(Matrix(1.0I, 3, 3))
  I33 = I3()
  ∂Ψuu(∇u) = _δδ_μ_3D(obj.μ) + _δδ_λ_3D(obj.λ)
  ∂Ψu(∇u) = ∂Ψuu(∇u) ⊙ (F(∇u) - I33)
  Ψ(∇u) = 0.5 * (F(∇u) - I33) ⊙ (∂Ψuu(∇u) ⊙ (F(∇u) - I33))
  return (Ψ, ∂Ψu, ∂Ψuu)
end

function (obj::NeoHookean3D)(::DerivativeStrategy{:analytic})
  F, H, J = _getKinematic(obj)
  Ψ(∇u) = obj.μ / 2 * tr((F(∇u))' * F(∇u)) - obj.μ * logreg(J(F(∇u))) + (obj.λ / 2) * (J(F(∇u)) - 1)^2 - 3.0 * (obj.μ / 2.0)
  ∂Ψ_∂J(∇u) = -obj.μ / J(F(∇u)) + obj.λ * (J(F(∇u)) - 1)
  ∂Ψu(∇u) = obj.μ * F(∇u) + ∂Ψ_∂J(∇u) * H(F(∇u))
  # I_ = TensorValue(Matrix(1.0I, 9, 9))
  I_ = I9()
  ∂Ψ2_∂J2(∇u) = obj.μ / (J(F(∇u))^2) + obj.λ
  ∂Ψuu(∇u) = obj.μ * I_ + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + ∂Ψ_∂J(∇u) * ×ᵢ⁴(F(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end

function (obj::MoneyRivlin3D)(::DerivativeStrategy{:analytic})
  F, H, J = _getKinematic(obj)
  Ψ(∇u) = obj.μ1 / 2 * tr((F(∇u))' * F(∇u)) + obj.μ2 / 2.0 * tr((H(F(∇u)))' * H(F(∇u))) - (obj.μ1 + 2 * obj.μ2) * logreg(J(F(∇u))) +
          (obj.λ / 2.0) * (J(F(∇u)) - 1)^2 - (3.0 / 2.0) * (obj.μ1 + obj.μ2)
  ∂Ψ_∂F(∇u) = obj.μ1 * F(∇u)
  ∂Ψ_∂H(∇u) = obj.μ2 * H(F(∇u))
  ∂Ψ_∂J(∇u) = -(obj.μ1 + 2.0 * obj.μ2) / J(F(∇u)) + obj.λ * (J(F(∇u)) - 1)
  ∂Ψ2_∂J2(∇u) = (obj.μ1 + 2.0 * obj.μ2) / (J(F(∇u))^2) + obj.λ
  ∂Ψu(∇u) = ∂Ψ_∂F(∇u) + ∂Ψ_∂H(∇u) × F(∇u) + ∂Ψ_∂J(∇u) * H(F(∇u))
  # I_ = TensorValue(Matrix(1.0I, 9, 9))
  I_ = I9()
  # ∂Ψuu(∇u) = obj.μ1 * I_ + obj.μ2 * (F(∇u) × (I_ × F(∇u))) + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + (I_ × (∂Ψ_∂H(∇u) + ∂Ψ_∂J(∇u) * F(∇u)))
  ∂Ψuu(∇u) = obj.μ1 * I_ + obj.μ2 * (F(∇u) × (I_ × F(∇u))) + ∂Ψ2_∂J2(∇u) * (H(F(∇u)) ⊗ H(F(∇u))) + ×ᵢ⁴(∂Ψ_∂H(∇u) + ∂Ψ_∂J(∇u) * F(∇u))
  return (Ψ, ∂Ψu, ∂Ψuu)
end

function (obj::ElectroMechModel)(strategy::DerivativeStrategy{:analytic})
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Mechano(strategy)
  Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.Mechano, obj.Electro)

  Ψ(∇u, ∇φ) = Ψm(∇u) + Ψem(∇u, ∇φ)
  ∂Ψu(∇u, ∇φ) = ∂Ψm_u(∇u) + ∂Ψem_u(∇u, ∇φ)
  ∂Ψφ(∇u, ∇φ) = ∂Ψem_φ(∇u, ∇φ)
  ∂Ψuu(∇u, ∇φ) = ∂Ψm_uu(∇u) + ∂Ψem_uu(∇u, ∇φ)
  ∂Ψφu(∇u, ∇φ) = ∂Ψem_φu(∇u, ∇φ)
  ∂Ψφφ(∇u, ∇φ) = ∂Ψem_φφ(∇u, ∇φ)

  return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
end

function (obj::ThermoMechModel)(strategy::DerivativeStrategy{:analytic})
  Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.Thermo(strategy)
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Mechano(strategy)
  Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.Mechano, obj.Thermo)

  f(δθ) = (obj.fθ(δθ)::Float64)
  df(δθ) = (obj.dfdθ(δθ)::Float64)

  Ψ(∇u, δθ) = f(δθ) * (Ψm(∇u)) + (Ψt(δθ) + Ψtm(∇u, δθ))
  ∂Ψu(∇u, δθ) = f(δθ) * (∂Ψm_u(∇u)) + ∂Ψtm_u(∇u, δθ)
  ∂Ψθ(∇u, δθ) = df(δθ) * (Ψm(∇u)) + ∂Ψtm_θ(∇u, δθ) + ∂Ψt_θ(δθ)

  ∂Ψuu(∇u, δθ) = f(δθ) * (∂Ψm_uu(∇u)) + ∂Ψtm_uu(∇u, δθ)
  ∂Ψθθ(∇u, δθ) = ∂Ψtm_θθ(∇u, δθ) + ∂Ψt_θθ(δθ)
  ∂Ψuθ(∇u, δθ) = df(δθ) * (∂Ψm_u(∇u)) + ∂Ψtm_uθ(∇u, δθ)

  η(∇u, δθ) = -∂Ψθ(∇u, δθ)

  return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ, η)
end


function (obj::ThermoMech_EntropicPolyconvex)(::DerivativeStrategy{:autodiff})
  Ψt, _, _ = obj.Thermo(DerivativeStrategy{:analytic}())
  Ψm, _, _ = obj.Mechano(DerivativeStrategy{:analytic}())

  θr = obj.Thermo.θr
  Cv = obj.Thermo.Cv
  α = obj.Thermo.α
  β = obj.β
  G = obj.G
  ϕ = obj.ϕ
  s = obj.s

  F, _, J = _getKinematic(obj.Mechano)
  I1, I2, I3 = _getInvariants_Isotropic(obj.Mechano)

  f(δθ) = (δθ + θr) / θr
  eᵣ(∇u) = α * (J(F(∇u)) - 1.0)
  L1(δθ) = (1 - β) * Ψt(δθ)
  L2(δθ) = Cv * θr * (1 - β) * G(f(δθ))
  L3(∇u, δθ) = -Cv * θr * β * s(I1(∇u), I2(∇u), I3(∇u)) * ϕ(f(δθ))

  Ψ(∇u, δθ) = f(δθ) * Ψm(∇u) + (1 - f(δθ)) * eᵣ(∇u) + L1(δθ) + L2(δθ) + L3(∇u, δθ)

  ∂Ψ_∂∇u(∇u, δθ)    = ForwardDiff.gradient(∇u -> Ψ(∇u, δθ), get_array(∇u))
  ∂Ψ_∂θ(∇u, δθ)     = ForwardDiff.derivative(δθ -> Ψ(get_array(∇u), δθ), δθ)
  ∂2Ψ_∂2∇u(∇u, δθ)  = ForwardDiff.hessian(∇u -> Ψ(∇u, δθ), get_array(∇u))
  ∂2Ψ_∂2θθ(∇u, δθ)  = ForwardDiff.derivative(δθ -> ∂Ψ_∂θ(get_array(∇u), δθ), δθ)
  ∂2Ψ_∂2∇uθ(∇u, δθ) = ForwardDiff.derivative(δθ -> ∂Ψ_∂∇u(get_array(∇u), δθ), δθ)

  ∂Ψu(∇u, δθ) = TensorValue(∂Ψ_∂∇u(∇u, δθ))
  ∂Ψθ(∇u, δθ) = ∂Ψ_∂θ(∇u, δθ)
  ∂Ψuu(∇u, δθ) = TensorValue(∂2Ψ_∂2∇u(∇u, δθ))
  ∂Ψθθ(∇u, δθ) = ∂2Ψ_∂2θθ(∇u, δθ)
  ∂Ψuθ(∇u, δθ) = TensorValue(∂2Ψ_∂2∇uθ(∇u, δθ))

  return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ)
end

function (obj::ThermoElectroMechModel)(strategy::DerivativeStrategy{:analytic})
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Mechano(strategy)
  Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.Thermo(strategy)
  Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.Mechano, obj.Electro)
  Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.Mechano, obj.Thermo)
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

  η(∇u, ∇φ, δθ) = -∂Ψθ(∇u, ∇φ, δθ)

  return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ, η)
end

function (obj::ThermalModel)(::DerivativeStrategy{:analytic})
  Ψ(δθ) = obj.Cv * (δθ - (δθ + obj.θr) * log((δθ + obj.θr) / obj.θr))
  ∂Ψθ(δθ) = -obj.Cv * log((δθ + obj.θr) / obj.θr)
  ∂Ψθθ(δθ) = -obj.Cv / (δθ + obj.θr)
  return (Ψ, ∂Ψθ, ∂Ψθθ)
end




end