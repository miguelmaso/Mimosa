using Pkg
Pkg.activate(".")
using Gridap
using GridapGmsh
using Gridap.TensorValues
using ForwardDiff
using BenchmarkTools
using LinearAlgebra
using Mimosa
using WriteVTK


# Initialisation result folder
mesh_file = "./models/mesh_platebeam_elec.msh"
result_folder = "./results/ex2/"
setupfolder(result_folder)

# Material parameters
const λ = 10.0
const μ = 1.0
const ε = 1.0
const autodif = true
solvertype = "direct"

# Kinematics
F(∇u) = one(∇u) + ∇u
J(F) = det(F)
H(F) = J(F) * inv(F)'
E(∇φ) = -∇φ
HE(∇u, ∇φ) = H(F(∇u)) * E(∇φ)
HEHE(∇u, ∇φ) = HE(∇u, ∇φ) ⋅ HE(∇u, ∇φ)
Ψm(∇u) = μ / 2 * tr((F(∇u))' * F(∇u)) - μ * logreg(J(F(∇u))) + (λ / 2) * (J(F(∇u)) - 1)^2
Ψe(∇u, ∇φ) = (-ε / (2 * J(F(∇u)))) * HEHE(∇u, ∇φ)
Ψ(∇u, ∇φ) = Ψm(∇u) + Ψe(∇u, ∇φ)


if autodif == true
  ∂Ψ_∂∇u(∇u, ∇φ) = ForwardDiff.gradient(∇u -> Ψ(∇u, get_array(∇φ)), get_array(∇u))
  ∂Ψ_∂∇φ(∇u, ∇φ) = ForwardDiff.gradient(∇φ -> Ψ(get_array(∇u), ∇φ), get_array(∇φ))
  ∂2Ψ_∂2∇φ(∇u, ∇φ) = ForwardDiff.hessian(∇φ -> Ψ(get_array(∇u), ∇φ), get_array(∇φ))
  ∂2Ψ_∂2∇u(∇u, ∇φ) = ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u, get_array(∇φ)), get_array(∇u))
  ∂2Ψ_∂2∇φ∇u(∇u, ∇φ) = ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇φ(∇u, get_array(∇φ)), get_array(∇u))

  ∂Ψu(∇u, ∇φ) = TensorValue(∂Ψ_∂∇u(∇u, ∇φ))
  ∂Ψφ(∇u, ∇φ) = VectorValue(∂Ψ_∂∇φ(∇u, ∇φ))
  ∂Ψuu(∇u, ∇φ) = TensorValue(∂2Ψ_∂2∇u(∇u, ∇φ))
  ∂Ψφφ(∇u, ∇φ) = TensorValue(∂2Ψ_∂2∇φ(∇u, ∇φ))
  ∂Ψφu(∇u, ∇φ) = TensorValue(∂2Ψ_∂2∇φ∇u(∇u, ∇φ))
else
  U_(∇u, ∇φ) = HE(∇u, ∇φ) ⊗ E(∇φ)
  ∂Ψu(∇u, ∇φ) = μ * F(∇u) + (-μ / J(F(∇u)) + λ * (J(F(∇u)) - 1.0) + (ε / (2 * J(F(∇u))^2.0)) * HEHE(∇u, ∇φ)) * H(F(∇u)) - (tr(U_(∇u, ∇φ)) * one(∇u) - U_(∇u, ∇φ)') * ε / J(F(∇u))
  ∂Ψφ(∇u, ∇φ) = (ε / J(F(∇u))) * (H(F(∇u))' * HE(∇u, ∇φ))
end


# model
#include("electro/model_electrobeam.jl")
# mesh_file = joinpath(dirname(@__FILE__), "ex2_mesh.msh")
model = GmshDiscreteModel(mesh_file)
model_file = joinpath(result_folder, "model")
writevtk(model, model_file)

labels = get_face_labeling(model)
add_tag_from_tags!(labels, "dirm_u0", [3])
add_tag_from_tags!(labels, "dire_mid", [1])
add_tag_from_tags!(labels, "dire_top", [2])
 
#Define reference FE (Q2/P1(disc) pair)
order = 1
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)

#Define test FESpaces
Vu = TestFESpace(model, reffeu, labels=labels, dirichlet_tags=["dirm_u0"], conformity=:H1)
Vφ = TestFESpace(model, reffeφ, labels=labels, dirichlet_tags=["dire_mid", "dire_top"], conformity=:H1)
V = MultiFieldFESpace([Vu, Vφ])

#Setup integration
degree = 2 * order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)

# # Weak form
function res((u, φ), (v, vφ))
  return ∫((∇(v)' ⊙ (∂Ψu ∘ (∇(u)', ∇(φ)))) + (∇(vφ)' ⋅ (∂Ψφ ∘ (∇(u)', ∇(φ))))) * dΩ
end

function jac((u, φ), (du, dφ), (v, vφ))
  return ∫(∇(v)' ⊙ (inner42 ∘ ((∂Ψuu ∘ (∇(u)', ∇(φ))), ∇(du)'))) * dΩ +
         ∫(∇(dφ) ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(u)', ∇(φ))), ∇(v)'))) * dΩ +
         ∫(∇(vφ)' ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(u)', ∇(φ))), ∇(du)'))) * dΩ +
         ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ))) ⋅ ∇(dφ))) * dΩ
end
 
# Setup non-linear solver
if solvertype == "amgcg"
using AlgebraicMultigrid: smoothed_aggregation
using AlgebraicMultigrid: aspreconditioner
pp(x) = aspreconditioner(smoothed_aggregation(x))
ls_ = IterativeSolver("cg"; Pl=pp, verbose=false, reltol=1e-7)
else
ls_= BackslashSolver()
end

nls = NLSolver(ls_;
  show_trace=true,
  method=:newton,
  iterations=20)

solver = FESolver(nls)
pvd_results = paraview_collection(result_folder*"results", append=false)

function NewtonRaphson(x0, φap, φ_max, loadinc, ndofm, cache)

  #Define trial FESpaces from Dirichlet values
  u0 = VectorValue(0.0, 0.0, 0.0)
  φ_bot = 0.0
  Uu = TrialFESpace(Vu, [u0])
  Uφ = TrialFESpace(Vφ, [φ_bot, φap])
  U = MultiFieldFESpace([Uu, Uφ])

  x0_old = copy(x0)

  #Update Dirichlet values
  uh = FEFunction(Uu, x0[1:ndofm])
  aφ(φ, vφ) = ∫(∇(vφ) ⋅ (∂Ψφ ∘ (∇(uh), ∇(φ)))) * dΩ
  lφ(vφ) = 0.0
  opφ = AffineFEOperator(aφ, lφ, Uφ, Vφ)
  φh = solve(opφ)
  x0[ndofm+1:end] = get_free_dof_values(φh)
  ph = FEFunction(U, x0)

  #FE problem
  op = FEOperator(res, jac, U, V)
  loadfact = round(φap / φ_max, digits=2)
  println("\n+++ Loadinc $loadinc:  φap $φap in loadfact $loadfact +++\n")

  cacheold = cache
  @time begin
  ph, cache = solve!(ph, solver, op, cache)
  end
  flag::Bool = (cache.result.f_converged || cache.result.x_converged)

  if (flag == true)
    pvd_results[loadinc] = createvtk(Ωₕ,result_folder * "_$loadinc.vtu", cellfields=["uh" => ph[1], "phi" => ph[2]],order=2)
    # writevtk(Ωₕ, "results/ex2/results_$(loadinc)", cellfields=["uh" => ph[1], "phi" => ph[2]])
    return get_free_dof_values(ph), cache, flag
  else
    return x0_old, cacheold, flag
  end
end

function SolveSteps()
  φ_max = 0.2
  nsteps = 30
  φ_inc = φ_max / nsteps

  x0 = zeros(Float64, num_free_dofs(V))
  ndofm::Int = num_free_dofs(Vu)


  cache = nothing
  φap = 0.0
  loadinc = 0
  maxbisect = 10
  nbisect = 0
  while (φap / φ_max) < 1.0 - 1e-6
    φap += φ_inc
    φap = min(φap, φ_max)
    x0, cache, flag = NewtonRaphson(x0, φap, φ_max, loadinc, ndofm, cache)
    if (flag == false)
      φap -= φ_inc
      φ_inc = φ_inc / 2
      nbisect += 1
    end
    if nbisect > maxbisect
      println("Maximum number of bisections reached")
      break
    end
    loadinc += 1
  end

  vtk_save(pvd_results)


end

@time SolveSteps()
