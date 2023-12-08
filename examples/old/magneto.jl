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

 

gradu = TensorValue(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
graduar = get_array(gradu)

Br = VectorValue(0.0, 0.0, 1.0)
 
Ba =VectorValue(0.0, 1.0, 0.0)
  
# Material parameters
const λ  = 10.0
const μ  = 1.0
const μ0 = 1.0
const autodif= true
# Kinematics
F(∇u) = one(∇u) + ∇u
J(F) = det(F)
H(F) = J(F) * inv(F)'
FBr(∇u,Br) =  F(∇u)*Br
FBr_Ba(∇u,Br,Ba) =  FBr(∇u,Br) ⋅ Ba
Ψmec(∇u) = μ / 2 * tr((F(∇u))' * F(∇u)) - μ * logreg(J(F(∇u))) + (λ / 2) * (J(F(∇u)) - 1)^2
Ψmag(∇u,Br,Ba) = -μ0*(FBr_Ba(∇u,Br,Ba))
Ψ(∇u,Br,Ba) = Ψmec(∇u) + Ψmag(∇u,Br,Ba)

∂Ψ_∂∇u(∇u,Br,Ba)       =  ForwardDiff.gradient(∇u -> Ψ(∇u,get_array(Br),get_array(Ba)), get_array(∇u))
∂2Ψ_∂2∇u(∇u,Br,Ba)     =  ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u,get_array(Br),get_array(Ba)), get_array(∇u))
 
∂Ψu(∇u,Br,Ba)        = TensorValue(∂Ψ_∂∇u(∇u,Br,Ba))
 ∂Ψuu(∇u,Br,Ba)      = TensorValue(∂2Ψ_∂2∇u(∇u,Br,Ba))
 
 

 

   # model
#include("electro/model_electrobeam.jl")

model = GmshDiscreteModel("benchmarks/electro/slimbeam_electro.msh")
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "dirm_u0",  [4])

 
#Define reference FE (Q2/P1(disc) pair)
order = 1
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

#Define test FESpaces
V = TestFESpace(model, reffe, labels=labels, dirichlet_tags=["dirm_u0"], conformity=:H1)

#Setup integration
degree = 2*order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)
 
 
# # Weak form
function res(u, v)
  return ∫((∇(v) ⊙ (∂Ψu ∘ (∇(u))))) * dΩ
end
  

function jac(u, du , v)
  return ∫(∇(v)  ⊙ (inner42∘((∂Ψuu ∘ (∇(u))), ∇(du)))) * dΩ
end

# Setup non-linear solver
nls = NLSolver(
    show_trace=true,
    method=:newton)

solver = FESolver(nls)
 
#function run(x0, φap, step, nsteps, ndofm ,cache)
function run(x0, cache)

  #Define trial FESpaces from Dirichlet values
  u0 = VectorValue(0.0, 0.0,0.0)
  U = TrialFESpace(V, [u0])
 
  #Update Dirichlet values
  uh = FEFunction(U, x0)
  #FE problem
  op = FEOperator(res,jac, U, V)
  #println("\n+++ Solving for φap $φap in step $step of $nsteps +++\n")
  uh, cache = solve!(uh, solver, op, cache)

   writevtk(Ωₕ, "results/results_$(lpad(step,3,'0'))", cellfields=["uh" => uh])

  return get_free_dof_values(uh), cache
end
#end


function runs()
#  φ_max = 0.1
#  φ_inc = 0.01
#  nsteps = ceil(Int, abs(φ_max) / φ_inc)

  x0 = zeros(Float64, num_free_dofs(V))

  cache = nothing
#  for step in 1:nsteps
#     φap = step * φ_max / nsteps
#     x0, cache = run(x0, φap, step, nsteps, ndofm ,cache)
     x0, cache = run(x0, cache)
#  end

end

@time runs()