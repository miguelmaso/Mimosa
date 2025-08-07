using Gridap.Arrays
using Gridap.TensorValues
using Mimosa.ConstitutiveModels
using Test
import Base:isapprox


function isapprox(A::TensorValue, B::AbstractArray; kwargs...)
    isapprox(get_array(A), B; kwargs...)
end

function isapprox(A::AbstractArray, B::TensorValue; kwargs...)
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

μ = 5.832e4
λ = 10*μ

@testset "LinearElasticity3D" begin
    model = LinearElasticity3D(λ=λ, μ=μ)
    testMechanicalModelDerivatives(model, F; tolerance=1e-14)
end

@testset "NeoHookean3D" begin
    model = NeoHookean3D(λ=λ, μ=μ)
    testMechanicalModelDerivatives(model, F; tolerance=1e-14)
end

@testset "MoneyRivlin3D" begin
    model = MoneyRivlin3D(λ=λ, μ1=μ, μ2=μ)
    testMechanicalModelDerivatives(model, F; tolerance=1e-14)
end



