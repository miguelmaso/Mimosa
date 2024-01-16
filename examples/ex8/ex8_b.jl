
using Gridap
using GridapEmbedded
using LinearAlgebra: tr
using Mimosa

function lame_parameters(E, ν)
  λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
  μ = E / (2 * (1 + ν))
  (λ, μ)
end

function main(; n, result_folder=nothing)

  # Background model
  L = VectorValue(1, 1, 1)
  partition = (n, n, n)
  pmin = Point(0.0, 0.0, 0.0)
  pmax = pmin + L
  periodicbs=true
  if periodicbs
   bgmodel = CartesianDiscreteModel(pmin, pmax, partition; isperiodic=(true, true, true))
  # Identify Dirichlet boundaries
  labeling = get_face_labeling(bgmodel)
  entity = num_entities(labeling) + 1
  labeling.d_to_dface_to_entity[1][1] = entity
  add_tag!(labeling, "support0", [entity])
  # Define geometry
  else
  bgmodel = CartesianDiscreteModel(pmin, pmax, partition)
  labeling = get_face_labeling(bgmodel)
  add_tag_from_tags!(labeling,"support0",[1,2,3,4,5,6,7,8])
  end
  Ω_bg = Triangulation(bgmodel)

  origin = Point(0.6, 0.5, 0.6)
  R = 0.2
  geo1 = sphere(R; x0=origin)
  origin2 = Point(0.3, 0.5, 0.3)
  geo2 = sphere(R; x0=origin2)


  geo3 = union(geo1,geo2,name="inclusion")
  geo4 = !(geo3, name = "matrix")

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
  # u1 = VectorValue(0.0,0.0,-0.001)

  # Cut the background model
  cutgeo = cut(bgmodel, union(geo3, geo4))

  # Setup interpolation mesh
  Ω1_act = Triangulation(cutgeo, ACTIVE, "inclusion")
  Ω2_act = Triangulation(cutgeo, ACTIVE, "matrix")

  # Setup integration meshes
  Ω1 = Triangulation(cutgeo, PHYSICAL, "inclusion")
  Ω2 = Triangulation(cutgeo, PHYSICAL, "matrix")
  Γ = EmbeddedBoundary(cutgeo, "inclusion", "matrix")

  # Setup normal vectors
  n_Γ = get_normal_vector(Γ)

  # Setup Lebesgue measures
  order = 1
  degree = 2 * order
  dΩ1 = Measure(Ω1, degree)
  dΩ2 = Measure(Ω2, degree)
  dΓ = Measure(Γ, degree)

  # Setup FESpace

  V1 = TestFESpace(Ω1_act,
    ReferenceFE(lagrangian, VectorValue{3,Float64}, order),
    conformity=:H1)
  V2 = TestFESpace(Ω2_act,
    ReferenceFE(lagrangian, VectorValue{3,Float64}, order),
    conformity=:H1,
    dirichlet_tags=["support0"])

  U1 = TrialFESpace(V1)
  U2 = TrialFESpace(V2, [u0])

  V = MultiFieldFESpace([V1, V2])
  U = MultiFieldFESpace([U1, U2])

  # Setup stabilization parameters

  meas_K1 = get_cell_measure(Ω1, Ω_bg)
  meas_K2 = get_cell_measure(Ω2, Ω_bg)
  meas_KΓ = get_cell_measure(Γ, Ω_bg)

  γ_hat = 20.0
  κ1 = CellField((E2 * meas_K1) ./ (E2 * meas_K1 .+ E1 * meas_K2), Ω_bg)
  κ2 = CellField((E1 * meas_K2) ./ (E2 * meas_K1 .+ E1 * meas_K2), Ω_bg)
  β = CellField((γ_hat * meas_KΓ) ./ (meas_K1 / E1 .+ meas_K2 / E2), Ω_bg)

  # Jump and mean operators for this formulation

  jump_u(u1, u2) = u1 - u2
  mean_t(u1, u2) = κ1 * (σ1 ∘ ε(u1)) + κ2 * (σ2 ∘ ε(u2))

  # Weak form

  a((u1, u2), (v1, v2)) =
    ∫(ε(v1) ⊙ (σ1 ∘ ε(u1))) * dΩ1 + ∫(ε(v2) ⊙ (σ2 ∘ ε(u2))) * dΩ2 +
    ∫(β * jump_u(v1, v2) ⋅ jump_u(u1, u2)
      -
      n_Γ ⋅ mean_t(u1, u2) ⋅ jump_u(v1, v2)
      -
      n_Γ ⋅ mean_t(v1, v2) ⋅ jump_u(u1, u2)) * dΓ

  f= VectorValue(0.0,0.0,0.005)

  l((v1, v2)) = ∫(v1 ⋅ f)dΩ1 + ∫(v2 ⋅ f)dΩ2


  # FE problem
  @time op = AffineFEOperator(a, l, U, V)
  @time uh1, uh2 = solve(op)
  uh = (uh1, uh2)
 
  # @show ∑(∫(∇(uh1))dΩ1)+ ∑(∫(∇(uh2))dΩ2)
  # Postprocess
  if result_folder !== nothing
    writevtk(Ω1,result_folder * "inclusion", cellfields=["uh" => uh1, "sigma" => σ1 ∘ ε(uh1)])
    writevtk(Ω2,result_folder * "matrix", cellfields=["uh" => uh2, "sigma" => σ2 ∘ ε(uh2)])
  end

end

folder = "./results/periodic_embed/"
setupfolder(folder)
@time main(n=11, result_folder=folder)
 
 