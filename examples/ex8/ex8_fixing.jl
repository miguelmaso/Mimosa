
using Pkg
Pkg.activate(".")
using Gridap
using GridapEmbedded
using LinearAlgebra: tr
using Mimosa
using Gridap.TensorValues
using WriteVTK
using ForwardDiff
 
 
result_folder = "./results/periodic_embed/"
setupfolder(result_folder)
# function main(; n, result_folder=nothing)
n = 51
# Material parameters
λ1 = 10.0
μ1 = 1.0
λ2 = 20.0
μ2 = 2.0

# Kinematics
F(∇u, Fm) = Fm + ∇u
J(F) = det(F)
H(F) = 0.5 * (F × F)
# Energy density
Ψ(∇u, Fm, μ, λ) = μ / 2 * tr((F(∇u, Fm))' * F(∇u, Fm)) - μ * log(J(F(∇u, Fm))) + (λ / 2) * (J(F(∇u, Fm)) - 1)^2
∂Ψ∂J(∇u, Fm, μ, λ) = -μ / J(F(∇u, Fm)) + λ * (J(F(∇u, Fm)) - 1)
∂2Ψ∂2J(∇u, Fm, μ, λ) = (λ + μ / (J(F(∇u, Fm)))^2)
∂3Ψ∂3J(∇u, Fm, μ, λ) = (-2 * μ / (J(F(∇u, Fm)))^3)
DΨ(∇u, Fm, μ, λ) = μ * F(∇u, Fm) + ∂Ψ∂J(∇u, Fm, μ, λ) * H(F(∇u, Fm))
D2Ψ(∇u, ∇v, Fm, μ, λ) = μ * ∇v + ∂Ψ∂J(∇u, Fm, μ, λ) * (F(∇u, Fm) × ∇v) + ∂2Ψ∂2J(∇u, Fm, μ, λ) * tr(H(F(∇u, Fm))' * ∇v) * H(F(∇u, Fm))

term1(∇u, ∇v, Fm) = (H(F(∇u, Fm)) ⊗  (∇v × F(∇u, Fm))) +  ((∇v × F(∇u, Fm)) ⊗ H(F(∇u, Fm))) + tr(H(F(∇u, Fm))' * ∇v) * cross_I4(F(∇u, Fm))
term2(∇u, ∇v, Fm) =  tr(H(F(∇u, Fm))' * ∇v) * H(F(∇u, Fm)) ⊗ H(F(∇u, Fm))
D3Ψ(∇u, ∇v, Fm, μ, λ) = ∂Ψ∂J(∇u, Fm, μ, λ) * cross_I4(∇v) + ∂2Ψ∂2J(∇u, Fm, μ, λ) * term1(∇u, ∇v, Fm) + ∂3Ψ∂3J(∇u, Fm, μ, λ) *term2(∇u, ∇v, Fm) 

# Background model
L = VectorValue(1, 1, 1)
partition = (n, n, n)
pmin = Point(0.0, 0.0, 0.0)
pmax = pmin + L
periodicbs = true
if periodicbs
  bgmodel = CartesianDiscreteModel(pmin, pmax, partition; isperiodic=(true, true, true))
  # Identify Dirichlet boundaries
  labeling = get_face_labeling(bgmodel)
  entity = num_entities(labeling) + 1
  labeling.d_to_dface_to_entity[1][1] = entity
  add_tag!(labeling, "support0", [entity])
else
  bgmodel = CartesianDiscreteModel(pmin, pmax, partition)
  labeling = get_face_labeling(bgmodel)
  add_tag_from_tags!(labeling, "support0", [1, 2, 3, 4, 5, 6, 7, 8])
end
Ω_bg = Triangulation(bgmodel)

# origin = Point(0.6, 0.5, 0.6)
origin = Point(0.5, 0.5, 0.5)
R = 0.2
geo1 = sphere(R; x0=origin, name="inclusion")
# origin2 = Point(0.3, 0.5, 0.3)
# geo2 = sphere(R; x0=origin2)

# geo3 = union(geo1, geo2, name="inclusion")
geo4 = !(geo1, name = "matrix")

# Dirichlet values
u0 = VectorValue(0, 0, 0)

# Cut the background model
# cutgeo = cut(bgmodel, union(geo3, geo4))
cutgeo = cut(bgmodel, union(geo1, geo4))

# Setup interpolation mesh
Ω1_act = Triangulation(cutgeo, ACTIVE, "inclusion")
Ω2_act = Triangulation(cutgeo, ACTIVE, "matrix")

# Setup integration meshes
Ω1 = Triangulation(cutgeo, PHYSICAL, "inclusion")
Ω2 = Triangulation(cutgeo, PHYSICAL, "matrix")
Γ = EmbeddedBoundary(cutgeo, "inclusion", "matrix")
writevtk(Ω1, result_folder * "Ω1")
writevtk(Ω2, result_folder * "Ω2")

# Setup normal vectors
n_Γ = get_normal_vector(Γ)
Q = TestFESpace(Ω1_act, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), conformity=:H1)

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

γ_hat = 2.0
κ1 = CellField((μ2 * meas_K1) ./ (μ2 * meas_K1 .+ μ1 * meas_K2), Ω_bg)
κ2 = CellField((μ1 * meas_K2) ./ (μ2 * meas_K1 .+ μ1 * meas_K2), Ω_bg)
β = CellField((γ_hat * meas_KΓ) ./ (meas_K1 / μ1 .+ meas_K2 / μ2), Ω_bg)

# Jump and mean operators for this formulation
jump_u(u1, u2) = u1 - u2

function Fmh(Λ::Float64)
  Fm = TensorValue(1.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  return CellField(Λ * (Fm - one(Fm)) + one(Fm), Ω_bg)
  # return  (x)->Λ * (Fm - one(Fm)) + one(Fm)

end

function mean_t(Λ::Float64)
  return (u1, u2) -> κ1 * (∂Ψ1u ∘ (∇(u1)', Fmh(Λ))) + κ2 * (∂Ψ2u ∘ (∇(u2)', Fmh(Λ)))
end


function res(Λ::Float64)
  return ((u1, u2), (v1, v2)) -> ∫((DΨ ∘ (∇(u1)', Fmh(Λ),μ1_, λ1_)) ⊙ ∇(v1)') * dΩ1 +
                                 ∫((DΨ ∘ (∇(u2)', Fmh(Λ),μ2_, λ2_)) ⊙ ∇(v2)') * dΩ2 +
                                 ∫(β * jump_u(v1, v2) ⋅ jump_u(u1, u2)) * dΓ
                                 
                                #  -
                                #    (κ1 * (DΨ ∘ (∇(u1)', Fmh(Λ),μ1_, λ1_)) ⊙ (jump_u(v1, v2) ⊗ n_Γ)) -
                                #    (κ2 * (DΨ ∘ (∇(u2)', Fmh(Λ),μ2_, λ2_)) ⊙ (jump_u(v1, v2) ⊗ n_Γ)) -
                                #    (κ1 * (D2Ψ ∘ (∇(u1)', ∇(v1)', Fmh(Λ),μ1_, λ1_)) ⊙ (jump_u(u1, u2) ⊗ n_Γ)) -
                                #    (κ2 * (D2Ψ ∘ (∇(u2)', ∇(v2)', Fmh(Λ),μ2_, λ2_)) ⊙ (jump_u(u1, u2) ⊗ n_Γ))) * dΓ
end



x0 = zeros(Float64, num_free_dofs(U))
uh = FEFunction(U, x0)


λ1_ = CellField(λ1, Ω_bg)
μ1_ = CellField(μ1, Ω_bg)
λ2_ = CellField(λ2, Ω_bg)
μ2_ = CellField(μ2, Ω_bg)




function jac(Λ::Float64)
  return ((u1, u2), (du1, du2), (v1, v2)) -> ∫((D2Ψ ∘ (∇(u1)', ∇(v1)', Fmh(Λ),μ1_, λ1_)) ⊙ ∇(du1)') * dΩ1 +
                                             ∫((D2Ψ ∘ (∇(u2)', ∇(v2)', Fmh(Λ),μ2_, λ2_)) ⊙ ∇(du2)') * dΩ2 +
                                             ∫(β * jump_u(v1, v2) ⋅ jump_u(du1, du2)  ) * dΓ 
                                             
                                            #  -
                                            #    (κ1 * (D2Ψ ∘ (∇(u1)', ∇(v1)', Fmh(Λ),μ1_, λ1_)) ⊙ (jump_u(du1, du2) ⊗ n_Γ)) -
                                            #    (κ2 * (D2Ψ ∘ (∇(u2)', ∇(v2)', Fmh(Λ),μ2_, λ2_)) ⊙ (jump_u(du1, du2) ⊗ n_Γ)) -
                                            #    (κ1 * (D2Ψ ∘ (∇(u1)', ∇(du1)', Fmh(Λ),μ1_, λ1_)) ⊙ (jump_u(v1, v2) ⊗ n_Γ)) -
                                            #    (κ2 * (D2Ψ ∘ (∇(u2)', ∇(du2)', Fmh(Λ),μ2_, λ2_)) ⊙ (jump_u(v1, v2) ⊗ n_Γ)) - 
                                            #    (∇(du1)') ⊙ (κ1 * (D3Ψ ∘ (∇(u1)', ∇(v1)', Fmh(Λ),μ1_, λ1_)) ⊙ (jump_u(u1, u2) ⊗ n_Γ)) -
                                            #    (∇(du2)') ⊙ (κ2 * (D3Ψ ∘ (∇(u2)', ∇(v2)', Fmh(Λ),μ2_, λ2_)) ⊙ (jump_u(u1, u2) ⊗ n_Γ)) ) * dΓ
end

# Setup non-linear solver
solvertype = "amgcg"
if solvertype == "amgcg"
  using AlgebraicMultigrid: smoothed_aggregation
  using AlgebraicMultigrid: aspreconditioner
  pp(x) = aspreconditioner(smoothed_aggregation(x))
  ls_ = IterativeSolver("cg"; Pl=pp, verbose=true, reltol=1e-7)
else
  ls_ = BackslashSolver()
end


nls = NLSolver(ls_;
  show_trace=true,
  method=:newton,
  iterations=25)

solver = FESolver(nls)
pvd_results_inclusion = paraview_collection(result_folder * "results", append=false)
pvd_results_matrix = paraview_collection(result_folder * "results", append=false)



function SolveStep(x0, Λ, cache)
  x0_old = copy(x0)
  cacheold = cache

  # Update FEFunction uh from vector u
  uh = FEFunction(U, x0)

  #Update Dirichlet values FE problem
  op = FEOperator(res(Λ), jac(Λ), U, V)
  uh, cache = solve!(uh, solver, op, cache)
   flag::Bool = (cache.result.f_converged || cache.result.x_converged)

  if (flag == true)
    return get_free_dof_values(uh), cache, flag
  else
    return x0_old, cacheold, flag
  end
end


function SolveSteps()
  Λ = 0.0
  nsteps = 10
  Λ_inc = 1.0 / nsteps

  x0 = zeros(Float64, num_free_dofs(V))

  cache = nothing
  loadinc = 0
  maxbisect = 10
  nbisect = 0
  while (Λ) < 1.0 - 1e-6
    Λ += Λ_inc
    Λ = min(Λ, 1.0)
    println("\n+++ Loadinc $loadinc:  Λ $Λ +++\n")
    x0, cache, flag = SolveStep(x0, Λ, cache)


    if (flag == false)
      Λ -= Λ_inc
      Λ_inc = Λ_inc / 2
      nbisect += 1
    else
      uh = FEFunction(U, x0)
      # FF = TensorValue(1.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
      # def(x) = Λ * (FF * x)
      # error("stop")

      # defh = interpolate_everywhere(def, V)

      pvd_results_inclusion[loadinc] = createvtk(Ω1, result_folder * "inclusion_$loadinc.vtu", cellfields=["uh" => uh[1]], order=2)
      pvd_results_matrix[loadinc] = createvtk(Ω2, result_folder * "matrix_$loadinc.vtu", cellfields=["uh" => uh[2]], order=2)
      loadinc += 1
    end
    if nbisect > maxbisect
      println("Maximum number of bisections reached")
      break
    end
  end


  vtk_save(pvd_results_inclusion)
  vtk_save(pvd_results_matrix)
end

@time SolveSteps()