using Gridap


@testset "Jacobian regularization" begin
    ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
    F = one(∇u) + ∇u
    J = det(F)
    @test J == 1.0149819999999996
    @test logreg(J; Threshold=0.01) == 0.014870878346353422  
  end

  @testset "NeoHookean3D" begin
    ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
    model = NeoHookean3D(3.0, 1.0)
    Ψ, ∂Ψu, ∂Ψuu = model(DerivativeStrategy{:analytic}())
    @test Ψ(∇u) == 0.0006083121396460722
    @test norm(∂Ψu(∇u)) == 0.099612127449168118
    @test norm(∂Ψuu(∇u)) == 12.073268944343628
  end

  @testset "MoneyRivlin3D" begin
    ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
    model = MoneyRivlin3D(3.0, 1.0, 2.0)
    Ψ, ∂Ψu, ∂Ψuu = model(DerivativeStrategy{:analytic}())
    @test Ψ(∇u) == 0.001598259078230413
    @test norm(∂Ψu(∇u)) == 0.24833325775972206
    @test norm(∂Ψuu(∇u)) == 30.36786840739546
  end