using Gridap
using GridapGmsh
using Gridap.TensorValues
import Base: *
using LineSearches: BackTracking
using ForwardDiff
using BenchmarkTools
using LinearAlgebra
using Mimosa 
 

@inline function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end

@inline function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end

 

# Material parameters
const λ = 10.0
const μ = 1.0
const autodif= true
# Kinematics
F(∇u) = one(∇u) + ∇u
J(F) = det(F)
H(F) = J(F) * inv(F)'
Ψ(∇u) = μ / 2 * tr((F(∇u))' * F(∇u)) - μ * log(J(F(∇u))) + (λ / 2) * (J(F(∇u)) - 1)^2
 
if autodif==true
∂Ψ_∂∇u(∇u)    = TensorValue(ForwardDiff.gradient(Ψ, get_array(∇u)))
∂2Ψ_∂2∇u(∇u)  = TensorValue(ForwardDiff.hessian(Ψ, get_array(∇u)))
else
 ∂Ψ_∂∇u(∇u) = μ*F(∇u)+(-μ/J(F(∇u))+λ*(J(F(∇u))-1.0))*H(F(∇u))
end

# gradu = TensorValue(0.8, 0.4, 0.7, 0.8, 0.4, 0.7, 0.8, 0.4, 0.7)
# @btime (∂2Ψ_∂2∇u($gradu))
# error("stop")

  # model
include("electro/mecmodel.jl")

model = GmshDiscreteModel("benchmarks/electro/mecmodel.msh") 
 
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "dirm_u0",  [4])
add_tag_from_tags!(labels, "force2", [5])
 
#Define reference FE (Q2/P1(disc) pair)
order = 1
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

#Define test FESpaces
V = TestFESpace(model, reffeu, labels=labels, dirichlet_tags=["dirm_u0"], conformity=:H1)

#Setup integration
degree = 2*order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)

neumanntags = ["force2"]
Γ = BoundaryTriangulation(model,tags=neumanntags)
dΓ = Measure(Γ,degree)


    # Setup non-linear solver
nls = NLSolver(
  show_trace=true,
  method=:newton)

 solver = FESolver(nls)


#Define trial FESpaces from Dirichlet values
u0 = VectorValue(0.0, 0.0,0.0)
U = TrialFESpace(V, [u0])
  
 

function run(x0::Vector{Float64}, fap, step, nsteps, cache)
 
  f(x) = VectorValue(0.0,-fap,0.0)
  res(u, v)  = ∫(∇(v) ⊙ (∂Ψ_∂∇u ∘ (∇(u)))) * dΩ - ∫(v⋅f)*dΓ 
  jac(u,du,v) = ∫(∇(v) ⊙ (inner42∘((∂2Ψ_∂2∇u ∘ (∇(u))), ∇(du)))) * dΩ

  op = FEOperator(res,jac, U, V)

  println("\n+++ Solving for fap $fap in step $step of $nsteps +++\n")

  uh = FEFunction(U,x0)
  
  uh, cache = solve!(uh, solver, op, cache )
   writevtk(Ωₕ, "results/results_$(lpad(step,3,'0'))", cellfields=["uh" => uh])

  return get_free_dof_values(uh)::Vector{Float64}, cache::Gridap.Algebra.NLSolversCache
end


function runs()
  f_max = 0.01
  f_inc = 0.001
  nsteps = ceil(Int, abs(f_max) / f_inc)
  ndof =num_free_dofs(V)::Int64 
  x0 = zeros(Float64, ndof)

  cache = nothing
  for step in 1:nsteps
     fap = step * f_max / nsteps
     x0, cache = run(x0, fap, step, nsteps, cache)
  end

end

   @time runs()
