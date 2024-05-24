using Gridap




@testset "LinearElasticity3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  modelLE = LinearElasticity3D(λ=3.0, μ=1.0)
  Ψ, ∂Ψu, ∂Ψuu = modelLE(DerivativeStrategy{:analytic}())
  @test (Ψ(∇u)) == 0.0006464999999999874
  @test norm(∂Ψu(∇u)) == 0.10157263410978287
  @test norm(∂Ψuu(∇u)) == 11.874342087037917
end


@testset "NeoHookean3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  model = NeoHookean3D(λ=3.0, μ=1.0)
  Ψ, ∂Ψu, ∂Ψuu = model(DerivativeStrategy{:analytic}())
  @test Ψ(∇u) == 0.0006083121396460722
  @test norm(∂Ψu(∇u)) == 0.099612127449168118
  @test norm(∂Ψuu(∇u)) == 12.073268944343628
end

@testset "MoneyRivlin3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  model = MoneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  Ψ, ∂Ψu, ∂Ψuu = model(DerivativeStrategy{:analytic}())
  @test Ψ(∇u) == 0.001598259078230413
  @test norm(∂Ψu(∇u)) == 0.24833325775972206
  @test norm(∂Ψuu(∇u)) == 30.36786840739546
end


@testset "ElectroMechano" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  modelMR = MoneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealDielectric(ε=4.0)
  modelelectro = ElectroMechModel( Mechano=modelMR, Electro=modelID)
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
  modelMR = MoneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealDielectric(ε=4.0)
  modelT = ThermalModel(Cv=1.0, θr= 1.0,α= 2.0)
  f(δθ::Float64)::Float64 = (δθ+1.0) / 1.0
  df(δθ::Float64)::Float64 = 1.0
  modelTEM = ThermoElectroMechModel(Thermo=modelT, Electro=modelID, Mechano=modelMR, fθ=f, dfdθ=df)
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
  θt = 3.4-1.0
  modelMR = MoneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelT = ThermalModel(Cv=1.0, θr= 1.0,α= 2.0)
  f(δθ::Float64)::Float64 = (δθ+1.0) / 1.0
  df(δθ::Float64)::Float64 = 1.0
  modelTM = ThermoMechModel(Thermo=modelT, Mechano=modelMR, fθ=f, dfdθ=df)
  Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = modelTM(Mimosa.DerivativeStrategy{:analytic}())

  @test (Ψ(∇u, θt)) == -2.190116215314799
  @test norm(∂Ψu(∇u, θt)) == 50.34457217400186
  @test norm(∂Ψθ(∇u, θt)) == 1.4033079344878807
  @test norm(∂Ψuu(∇u, θt)) == 132.85408867418602
  @test norm(∂Ψθθ(∇u, θt)) == 0.29411764705882354
  @test norm(∂Ψuθ(∇u, θt)) == 21.074087978716364


end




@testset "ThermoMech_EntropicPolyconvex" begin

  ∇u  =  1e-1*TensorValue(1,2,3,4,5,6,7,8,9)
  θt     =  21.6
  modmec  = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=1.0, ρ=1.0)
  modterm = ThermalModel(Cv=3.4, θr=2.2, α=1.2, κ=1.0)
  β=0.7
  G(x)    =  x*(log(x) - 1.0) - 4/3*x^(3/2) + 2*x +  1/3
  γ₁      =  0.5
  γ₂      =  0.5
  γ₃      =  0.5
  s(I1,I2,I3) =  1/3*((I1/3.0)^γ₁ + (I2/3.0)^γ₂ + I3^γ₃)
  ϕ(x)        =  2.0*(x+1.0)*log(x+1.0) - 2.0*x*(1+log(2)) + 2.0*(1 - log(2))
  consmodel = ThermoMech_EntropicPolyconvex(Thermo=modterm, Mechano=modmec,   β=β, G=G, ϕ=ϕ, s=s )

  Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = consmodel(DerivativeStrategy{:autodiff}())

  @test (Ψ(∇u, θt)) == -129.4022076861008
  @test norm(∂Ψu(∇u, θt)) == 437.9269386687991
  @test norm(∂Ψθ(∇u, θt)) == 13.97666807099424
  @test norm(∂Ψuu(∇u, θt)) == 2066.7910102392775
  @test norm(∂Ψθθ(∇u, θt)) == 0.46689338540182707
  @test norm(∂Ψuθ(∇u, θt)) == 14.243050132210923

end
