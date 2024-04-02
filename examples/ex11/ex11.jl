
using Gridap
using Gridap.FESpaces
using Mimosa
using GridapGmsh
using Gridap.Geometry
using LinearAlgebra: tr


function lame_parameters(E, ν)
    λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
    μ = E / (2 * (1 + ν))
    (λ, μ)
  end
  
result_folder = "./results/periodic/"
setupfolder(result_folder)

model = GmshDiscreteModel("./examples/ex11/RVEsfr2.msh")
model_file = joinpath(result_folder, "model")
 
Ω0 = Triangulation(model,tags="Phase0")
Ω1 = Triangulation(model,tags="Phase1")
Ω = Triangulation(model )

 # Material parameters inclusion
 E1 = 1
 ν1 = 0.2
 λ1, μ1 = lame_parameters(E1, ν1)
 σ1(ε) = λ1 * tr(ε) * one(ε) + 2 * μ1 * ε

 # Material parameters matrix
 E2 = 1
 ν2 = 0.2
 λ2, μ2 = lame_parameters(E2, ν2)
 σ2(ε) = λ2 * tr(ε) * one(ε) + 2 * μ2 * ε

   # Dirichlet values
   u0 = VectorValue(0, 0, 0)

    # Setup Lebesgue measures
  order = 1
  degree = 2 * order
  dΩ0 = Measure(Ω0, degree)
  dΩ1 = Measure(Ω1, degree)
  dΩ = Measure(Ω, degree)


  # Setup FESpace
 
    V = TestFESpace(Ω,
    ReferenceFE(lagrangian, VectorValue{3,Float64}, order),
    conformity=:H1,
    dirichlet_tags=["Corners"])

    U = TrialFESpace(V, [u0])
  

  # Weak form

  a(u, v ) =
  ∫(ε(v) ⊙ (σ1 ∘ ε(u))) * dΩ0 + ∫(ε(v) ⊙ (σ2 ∘ ε(u))) * dΩ1 

f= VectorValue(0.0,0.0,0.5)

l(v) = ∫(v ⋅ f)dΩ0 + ∫(v ⋅ f)dΩ1

  # FE problem
  @time op = AffineFEOperator(a, l, U, V)
  @time uh = solve(op)
 
  @show ∑(∫(∇(uh))dΩ) 

writevtk(Ω,result_folder * "resultssinperiodic",cellfields=["uh"=>uh])
  
 