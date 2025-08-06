module WeakForms

using Gridap
using Mimosa.TensorAlgebra
using Mimosa.PhysicalModels

export residual
export jacobian
export mass_term
export residual_Neumann
export (+)

import Base: +

# Coupling management
# ===================

function (+)(::Nothing, b::Gridap.CellData.DomainContribution)
b
end
function (+)(b::Gridap.CellData.DomainContribution, ::Nothing)
b
end
function (+)(a::Nothing, ::Nothing)
    a
end

# ===================
# Mechanics
# ===================

function residual(::Type{Mechano}, u, v, ∂Ψu, dΩ)
    ∫(∇(v)' ⊙ (∂Ψu ∘ (∇(u)')))dΩ
end

function jacobian(::Type{Mechano}, u, du, v, ∂Ψuu, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)')) ⊙ (∇(du)')))dΩ
end

function mass_term(u, v, Coeff, dΩ)
    ∫(Coeff* (u⋅v))dΩ
end


function residual_Neumann(t, v, bc, dΓ)
    # TODO 1: Asignar res=nothing está mal. No se puede aplicar el operador +
    # TODO 2: La definición de la inegral no está bien, no sé cómo se pasan los argumentos a la función de 'fuerza'
    # TODO 3: Probar mapreduce
    # mapreduce(val -> ∫(val(t)(v))dΓ, +, bc.values)
    res = nothing
    for i in eachindex(bc.tags)
        res += ∫(bc.values[i](t)(v))dΓ
    end
    return res
end


# ===================
# ThermoElectroMech
# ===================

# Stagered strategy
# -----------------
function residual(::Type{ThermoElectroMechano}, ::Type{Mechano}, (u, φ, θ), v, ∂Ψu, dΩ)
    return ∫(∇(v)' ⊙ (∂Ψu ∘ (∇(u)', ∇(φ), θ)))dΩ
end

function residual(::Type{ThermoElectroMechano}, ::Type{Electro}, (u, φ, θ), vφ, ∂Ψφ, dΩ)
    return ∫(∇(vφ)' ⋅ (∂Ψφ ∘ (∇(u)', ∇(φ), θ)))dΩ
end

function residual(::Type{ThermoElectroMechano}, ::Type{Thermo}, (u, φ, θ), vθ, κ, dΩ)
    return ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end

function jacobian(::Type{ThermoElectroMechano}, ::Type{Mechano}, (u, φ, θ), du, v, ∂Ψuu, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', ∇(φ), θ)) ⊙ (∇(du)')))dΩ
end

function jacobian(::Type{ThermoElectroMechano}, ::Type{Electro}, (u, φ, θ), dφ, vφ, ∂Ψφφ, dΩ)
    ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ), θ)) ⋅ ∇(dφ)))dΩ
end

function jacobian(::Type{ThermoElectroMechano}, ::Type{Thermo}, dθ, vθ, κ, dΩ)
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(::Type{ThermoElectroMechano}, ::Type{Thermo}, (u, φ, θ), dθ, vθ, κ, dΩ)
    ∫((κ ∘ (u, φ, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(::Type{ThermoElectroMechano}, ::Type{ElectroMechano}, (u, φ, θ), (du, dφ), (v,vφ), ∂Ψφu, dΩ)
    ∫(∇(dφ) ⋅ ((∂Ψφu ∘ (∇(u)', ∇(φ), θ)) ⊙ (∇(v)')))dΩ + 
    ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (∇(u)', ∇(φ), θ)) ⊙ (∇(du)')))dΩ 
end

function jacobian(::Type{ThermoElectroMechano}, ::Type{ThermoMechano}, (u, φ, θ), dθ, v, ∂Ψuθ, dΩ)
    ∫(∇(v)' ⊙ (∂Ψuθ ∘ (∇(u)', ∇(φ), θ)) * dθ)dΩ 
end

function jacobian(::Type{ThermoElectroMechano}, ::Type{ThermoElectro}, (u, φ, θ), dθ, vφ, ∂Ψφθ, dΩ)
    ∫(∇(vφ) ⋅ ((∂Ψφθ ∘ (∇(u)', ∇(φ), θ)) * dθ))dΩ
end

# Monolithic strategy
# -------------------
function residual(::Type{ThermoElectroMechano}, (u, φ, θ), (v, vφ, vθ), (∂Ψu, ∂Ψφ), κ, dΩ)
    residual(ThermoElectroMechano, Mechano, (u, φ, θ), v, ∂Ψu, dΩ) +
    residual(ThermoElectroMechano, Electro, (u, φ, θ), vφ, ∂Ψφ, dΩ) +
    residual(ThermoElectroMechano, Thermo, (u, φ, θ), vθ, κ, dΩ)
end

function jacobian(::Type{ThermoElectroMechano},   (u, φ, θ), (du, dφ, dθ), (v, vφ, vθ), (∂Ψuu, ∂Ψφφ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ), κ, dΩ)
    jacobian(ThermoElectroMechano, Mechano, (u, φ, θ), du, v, ∂Ψuu, dΩ)+
    jacobian(ThermoElectroMechano, Electro, (u, φ, θ), dφ, vφ, ∂Ψφφ, dΩ)+
    jacobian(ThermoElectroMechano, Thermo, dθ, vθ, κ, dΩ)+
    jacobian(ThermoElectroMechano, ElectroMechano, (u, φ, θ), (du, dφ), (v,vφ), ∂Ψφu, dΩ)+
    jacobian(ThermoElectroMechano, ThermoMechano, (u, φ, θ), dθ, v, ∂Ψuθ, dΩ)+
    jacobian(ThermoElectroMechano, ThermoElectro, (u, φ, θ), dθ, vφ, ∂Ψφθ, dΩ)
end


# ===================
# ThermoMech
# ===================

# Stagered strategy
# -----------------
function residual(::Type{ThermoMechano}, ::Type{Mechano}, (u, θ), v, ∂Ψu, dΩ)
    ∫(∇(v)' ⊙ (∂Ψu ∘ (∇(u)', θ)))dΩ
end

function residual(::Type{ThermoMechano}, ::Type{Thermo}, (u, θ), vθ, κ, dΩ)
    ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end

function jacobian(::Type{ThermoMechano}, ::Type{Mechano}, (u, θ), du, v, ∂Ψuu, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', θ)) ⊙ (∇(du)')))dΩ
end

function jacobian(::Type{ThermoMechano}, ::Type{Thermo}, dθ, vθ, κ, dΩ)
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(::Type{ThermoMechano}, ::Type{Thermo}, (u, θ), dθ, vθ, κ, dΩ)
    ∫((κ ∘ (u, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(::Type{ThermoMechano}, ::Type{ThermoMechano}, (u, θ), (du, dθ), v, ∂Ψuθ, dΩ)
    ∫(∇(v)' ⊙ (∂Ψuθ ∘ (∇(u)', θ)) * dθ)dΩ 
end

# Monolithic strategy
# -------------------
function residual(::Type{ThermoMechano},  (u, θ), (v, vθ), ∂Ψu, κ, dΩ)
    residual(ThermoMechano, Mechano, (u, θ), v, ∂Ψu, dΩ) +
    residual(ThermoMechano, Thermo, (u, θ), vθ, κ, dΩ)
end

function jacobian(::Type{ThermoMechano},  (u, θ), (du, dθ), (v, vθ), (∂Ψuu, ∂Ψuθ), κ, dΩ)
    jacobian(ThermoMechano, Mechano, (u, θ), du, v, ∂Ψuu, dΩ)+
    jacobian(ThermoMechano, Thermo, dθ, vθ, κ, dΩ)+
    jacobian(ThermoMechano, ThermoMechano, (u, θ), (du, dθ), v, ∂Ψuθ, dΩ)
end

# ===================
# ElectroMechanics
# ===================

# Stagered strategy
# -----------------

function residual(::Type{ElectroMechano}, ::Type{Mechano}, (u, φ), v, ∂Ψu, dΩ)
    ∫((∇(v)' ⊙ (∂Ψu ∘ (∇(u)', ∇(φ)))))dΩ
end

function residual(::Type{ElectroMechano}, ::Type{Electro}, (u, φ), vφ, ∂Ψφ, dΩ)
    ∫((∇(vφ) ⋅ (∂Ψφ ∘ (∇(u)', ∇(φ)))))dΩ
end

function jacobian(::Type{ElectroMechano}, ::Type{Mechano}, (u, φ), du, v, ∂Ψuu, dΩ)
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', ∇(φ))) ⊙ (∇(du)')))dΩ
end

function jacobian(::Type{ElectroMechano}, ::Type{Electro}, (u, φ), dφ, vφ, ∂Ψφφ, dΩ)
    ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ))) ⋅ ∇(dφ)))dΩ
end

function jacobian(::Type{ElectroMechano}, ::Type{ElectroMechano}, (u, φ), (du, dφ), (v, vφ), ∂Ψφu, dΩ)
    ∫(∇(dφ) ⋅ ((∂Ψφu ∘ (∇(u)', ∇(φ))) ⊙ (∇(v)')))dΩ +
    ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (∇(u)', ∇(φ))) ⊙ (∇(du)')))dΩ 
end


# Monolithic strategy
# -------------------

function residual(::Type{ElectroMechano},   (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), dΩ)
    residual(ElectroMechano, Mechano, (u, φ), v, ∂Ψu, dΩ) +
    residual(ElectroMechano, Electro, (u, φ), vφ, ∂Ψφ, dΩ)
end


function jacobian(::Type{ElectroMechano},   (u, φ), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)
    jacobian(ElectroMechano, Mechano, (u, φ), du, v, ∂Ψuu, dΩ)+
    jacobian(ElectroMechano, Electro, (u, φ), dφ, vφ, ∂Ψφφ, dΩ)+
    jacobian(ElectroMechano, ElectroMechano, (u, φ), (du, dφ), (v, vφ), ∂Ψφu, dΩ)
end


end