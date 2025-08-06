using Gridap.TensorValues
using Mimosa.ConstitutiveModels
using StaticArrays
using Test
import Base:isapprox


function isapprox(A::TensorValue, B::StaticArray; kwargs...)
    isapprox(get_array(A), B; kwargs...)
end

function isapprox(A::StaticArray, B::TensorValue; kwargs...)
    isapprox(A, get_array(B); kwargs...)
end

function isapprox(A::TensorValue, B::TensorValue; kwargs...)
    isapprox(get_array(A), get_array(B); kwargs...)
end


"""
    testMechanicalModelDerivatives

Check the analytical expressions match the numerical ones from autodiff.
The check is performed for the energy density, its first derivative (i.e. stress) and the sedond derivative (Jacobian).

# Arguments
- `model::Mechano`
- `kinematic`: The kinematic definition, F or C
- `args`: Extra arguments to pass to the physical model, such as `StressTensor{:SecondPiola}()`
- `tolerance`: the relative tolerance
- `debug=:nothing`: it can be `:energy`, `:stress` or `:jacobian`
"""
function testMechanicalModelDerivatives(model::Mechano, kinematic, args...; tolerance=1e-12, debug=:nothing)
    Ψa, ∂Ψa∂u, ∂Ψa∂uu = model(DerivativeStrategy{:analytic}(), args...)
    Ψd, ∂Ψd∂u, ∂Ψd∂uu = model(DerivativeStrategy{:autodiff}(), args...)
    @test isapprox(Ψa(kinematic), Ψd(kinematic), rtol=tolerance)
    @test isapprox(∂Ψa∂u(kinematic), ∂Ψd∂u(kinematic), rtol=tolerance)
    @test isapprox(∂Ψa∂uu(kinematic), ∂Ψd∂uu(kinematic), rtol=tolerance)
    if debug == :energy
        @show Ψa(kinematic)
        @show Ψd(kinematic)
    elseif debug == :stress
        @show ∂Ψa∂uu(kinematic)
        @show ∂Ψd∂uu(kinematic)
    elseif debug == :jacobian
        @show ∂Ψa∂uu(kinematic)
        @show ∂Ψd∂uu(kinematic)
    end
end


F = TensorValue(0.01+1.0, 0.02, 0.03, 0.04, 0.05+1.0, 0.06, 0.07, 0.08, 0.09+1.0)
C = F' * F
# using Gridap.Arrays
# C = get_array(F' * F)  # TODO: Should we move to use ALWAYS TensorValue instead of SMatrix?????????



using Gridap.Arrays
using ForwardDiff
using Mimosa.TensorAlgebra
function _getKinematic(::Mechano)
  F(∇u) = one(∇u) + ∇u
  J(F) = det(F)
  H(F) = J(F) * inv(F)'
  return (F, H, J)
end
struct IncompressibleNeoHookean3D_2dP <: Mechano
  μ::Float64
end
function (obj::IncompressibleNeoHookean3D_2dP)(::DerivativeStrategy{:autodiff})
    Ψ(Ce)      = obj.μ / 2 * tr(Ce) * (det(Ce))^(-1 / 3)
    Se(Ce)     = 2 * ForwardDiff.gradient(Ce -> Ψ(Ce), get_array(Ce))
    ∂Se∂Ce(Ce) = ForwardDiff.jacobian(Ce -> Se(Ce), get_array(Ce))
    return (Ψ, Se, ∂Se∂Ce)
end
function (obj::IncompressibleNeoHookean3D_2dP)(::DerivativeStrategy{:analytic})
    _, H, J = _getKinematic(obj)
    μ = obj.μ
    I3__ = I3()
    Ψ(Ce) = μ / 2 * tr(Ce) * (det(Ce))^(-1 / 3)
    ∂Ψ∂Ce(Ce) =  μ / 2 * I3__ * (det(Ce))^(-1 / 3)
    ∂Ψ∂dCe(Ce) = - μ / 6 * tr(Ce) * (det(Ce))^(-4 / 3)
    Se(Ce)  = 2 * (∂Ψ∂Ce(Ce) + ∂Ψ∂dCe(Ce) * H(Ce))
    ∂2Ψ∂CedCe(Ce) =  - μ / 6 * I3__ * (det(Ce))^(-4 / 3)
    ∂2Ψ∂2dCe(Ce) =  2*μ / 9 * tr(Ce) * (det(Ce))^(-7 / 3)
    ∂Se∂Ce(Ce) = 2 *  (∂2Ψ∂2dCe(Ce) * (H(Ce) ⊗ H(Ce)) + ∂2Ψ∂CedCe(Ce) ⊗ H(Ce) + H(Ce) ⊗ ∂2Ψ∂CedCe(Ce) + ∂Ψ∂dCe(Ce) * ×ᵢ⁴(Ce))
    return (Ψ, Se, ∂Se∂Ce)
end


@testset "NeoHookean3D" begin
    μ = 5.832e4
    λ = 10*μ
    model = NeoHookean3D(λ=λ, μ=μ)
    testMechanicalModelDerivatives(model, F; tolerance=1e-14)
end

@testset "IncompressibleNeoHookean3D_2dP" begin
    μ = 5.832e4
    model = IncompressibleNeoHookean3D_2dP(μ)
    testMechanicalModelDerivatives(model, C; tolerance=1e-14)
end

@testset "IncompressibleNeoHookean3D" begin
    μ = 5.832e4
    model = IncompressibleNeoHookean3D(μ)
    testMechanicalModelDerivatives(model, F, StressTensor{:FirstPiola}(); tolerance=1e-14)
    testMechanicalModelDerivatives(model, C, StressTensor{:SecondPiola}(); tolerance=1e-14)
end




