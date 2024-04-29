using Gridap


@testset "Jacobian regularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  F = one(∇u) + ∇u
  J = det(F)
  @test J == 1.0149819999999996
  @test logreg(J; Threshold=0.01) == 0.014870878346353422
end

@testset "LinearElasticity3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  modelLE = LinearElasticity3D(3.0, 1.0)
  Ψ, ∂Ψu, ∂Ψuu = modelLE(DerivativeStrategy{:analytic}())
  @test (Ψ(∇u)) == 0.0006464999999999874
  @test norm(∂Ψu(∇u)) == 0.10157263410978287
  @test norm(∂Ψuu(∇u)) == 11.874342087037917
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


@testset "ElectroMech" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  modelMR = MoneyRivlin3D(3.0, 1.0, 2.0)
  modelID = IdealDielectric(4.0)
  modelelectro = ElectroMech(modelMR, modelID)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelelectro(DerivativeStrategy{:analytic}())
  @test Ψ(∇u, ∇φ) == -27.514219755428428
  @test norm(∂Ψu(∇u, ∇φ)) == 47.42294370458073
  @test norm(∂Ψφ(∇u, ∇φ)) == 14.707913034885005
  @test norm(∂Ψuu(∇u, ∇φ)) == 131.10069227603947
  @test norm(∂Ψφu(∇u, ∇φ)) == 39.03656526472973
  @test norm(∂Ψφφ(∇u, ∇φ)) == 6.964428025226914
end


@testset "TermoElectroMech" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  θt = 3.4-1.0
  modelMR = MoneyRivlin3D(3.0, 1.0, 2.0)
  modelID = IdealDielectric(4.0)
  modelT = ThermalModel(1.0, 1.0, 2.0)
  f(δθ::Float64)::Float64 = (δθ+1.0) / 1.0
  df(δθ::Float64)::Float64 = 1.0
  modelTEM = ThermoElectroMech(modelT, modelID, modelMR, f, df)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ = modelTEM(DerivativeStrategy{:analytic}())
  @test (Ψ(∇u, ∇φ, θt)) == -95.74389746463744
  @test norm(∂Ψu(∇u, ∇φ, θt)) == 185.1315441384458
  @test norm(∂Ψφ(∇u, ∇φ, θt)) == 50.00690431860902
  @test norm(∂Ψθ(∇u, ∇φ, θt)) == 28.91912594899454
  @test norm(∂Ψuu(∇u, ∇φ, θt)) == 429.9957659123366
  @test norm(∂Ψφφ(∇u, ∇φ, θt)) == 23.679055285771508
  @test norm(∂Ψθθ(∇u, ∇φ, θt)) == 0.29411764705882354
  @test norm(∂Ψφu(∇u, ∇φ, θt)) == 132.7243219000811
  @test norm(∂Ψuθ(∇u, ∇φ, θt)) == 58.281073490042175
  @test norm(∂Ψφθ(∇u, ∇φ, θt)) == 14.707913034885005
end




@testset "TermoMech" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  θt = 3.4-1.0
  modelMR = MoneyRivlin3D(3.0, 1.0, 2.0)
  modelT = ThermalModel(1.0, 1.0, 2.0)
  f(δθ::Float64)::Float64 = (δθ+1.0) / 1.0
  df(δθ::Float64)::Float64 = 1.0
  modelTM = ThermoMech(modelT, modelMR, f, df)
  Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = modelTM(Mimosa.DerivativeStrategy{:analytic}())

  @test (Ψ(∇u, θt)) == -2.190116215314799
  @test norm(∂Ψu(∇u, θt)) == 50.34457217400186
  @test norm(∂Ψθ(∇u, θt)) == 1.4033079344878807
  @test norm(∂Ψuu(∇u, θt)) == 132.85408867418602
  @test norm(∂Ψθθ(∇u, θt)) == 0.29411764705882354
  @test norm(∂Ψuθ(∇u, θt)) == 21.074087978716364


end