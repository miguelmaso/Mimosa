using Gridap
using GridapGmsh
using Gridap.TensorValues
import Base: *
using LineSearches: BackTracking
using ForwardDiff
using BenchmarkTools
using LinearAlgebra
using Mimosa
#  @inline function ⊗(Ten::VectorValue{2,Float64})
#  return TensorValue(Ten.data[1]^2, Ten.data[1] * Ten.data[2], Ten.data[1] * Ten.data[2], Ten.data[2]^2)
#  end
 
# @inline function outer(Ten1 ,Ten2 )
#      TensorValue(Ten1.data[1]*Ten2.data[1], 
#                  Ten1.data[2]*Ten2.data[1],
#                  Ten1.data[1]*Ten2.data[2], 
#                  Ten1.data[2]*Ten2.data[2])
#  end
#  const ⊗ = outer



@inline function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end

@inline function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end

  

gradu = TensorValue(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
graduar = get_array(gradu) 

gradp = VectorValue(0.1, 0.2, 0.3)
gradpar = get_array(gradu) 
 
# gradu = TensorValue(0.8, 0.4, 0.7, 0.5)
# graduar = get_array(gradu)


# gradp = VectorValue(0.8, 0.7)
# gradpar = get_array(gradp)


# Material parameters
const λ = 10.0
const μ = 1.0
const ε = 1.0
const autodif= true
# Kinematics
F(∇u) = one(∇u) + ∇u
J(F) = det(F)
H(F) = J(F) * inv(F)'
E(∇φ)=-∇φ
HE(∇u, ∇φ) =  H(F(∇u))*E(∇φ)
HEHE(∇u, ∇φ) =  HE(∇u, ∇φ) ⋅ HE(∇u, ∇φ)
Ψm(∇u) = μ / 2 * tr((F(∇u))' * F(∇u)) - μ * logreg(J(F(∇u))) + (λ / 2) * (J(F(∇u)) - 1)^2
Ψe(∇u, ∇φ) = (-ε / (2 * J(F(∇u)))) * HEHE(∇u, ∇φ)
Ψ(∇u, ∇φ) = Ψm(∇u) + Ψe(∇u, ∇φ)

if autodif==true
∂Ψ_∂∇u(∇u, ∇φ)       =  ForwardDiff.gradient(∇u -> Ψ(∇u, get_array(∇φ)), get_array(∇u))
∂Ψ_∂∇φ(∇u, ∇φ)       =  ForwardDiff.gradient(∇φ -> Ψ(get_array(∇u), ∇φ), get_array(∇φ))
∂2Ψ_∂2∇φ(∇u, ∇φ)     =  ForwardDiff.hessian(∇φ -> Ψ(get_array(∇u), ∇φ), get_array(∇φ))
∂2Ψ_∂2∇u(∇u, ∇φ)     =  ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u, get_array(∇φ)), get_array(∇u))
∂2Ψ_∂2∇φ∇u(∇u, ∇φ)   =  ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇φ(∇u, get_array(∇φ)), get_array(∇u))

∂Ψu(∇u, ∇φ)       = TensorValue(∂Ψ_∂∇u(∇u, ∇φ))
∂Ψφ(∇u, ∇φ)       = VectorValue(∂Ψ_∂∇φ(∇u, ∇φ))
∂Ψuu(∇u, ∇φ)      = TensorValue(∂2Ψ_∂2∇u(∇u, ∇φ))
∂Ψφφ(∇u, ∇φ)      = TensorValue(∂2Ψ_∂2∇φ(∇u, ∇φ))
∂Ψφu(∇u, ∇φ)      = TensorValue(∂2Ψ_∂2∇φ∇u(∇u, ∇φ))
else
 U_(∇u,∇φ)=HE(∇u, ∇φ) ⊗ E(∇φ) 
 ∂Ψu(∇u, ∇φ) = μ*F(∇u)+(-μ/J(F(∇u))+λ*(J(F(∇u))-1.0)+(ε/(2*J(F(∇u))^2.0))*HEHE(∇u, ∇φ))*H(F(∇u))-(tr(U_(∇u,∇φ))*one(∇u)-U_(∇u,∇φ)')*ε/J(F(∇u))
 ∂Ψφ(∇u, ∇φ) = (ε/J(F(∇u)))*(H(F(∇u))'*HE(∇u, ∇φ) )
end
 

# @btime (∂Ψu($gradu,$gradp))
# @btime (∂Ψφ($gradu,$gradp))
# @btime (∂Ψuu($gradu,$gradp))
# @btime (∂Ψφφ($gradu,$gradp))
# @btime (∂Ψφu($gradu,$gradp))

   # model
include("electro/model_electrobeam.jl")

model = GmshDiscreteModel("benchmarks/electro/slimbeam_electro.msh") 
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "dirm_u0",  [4])
add_tag_from_tags!(labels, "dire_mid", [1])
add_tag_from_tags!(labels, "dire_top", [2])
 
#Define reference FE (Q2/P1(disc) pair)
order = 2
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)

#Define test FESpaces
Vu = TestFESpace(model, reffeu, labels=labels, dirichlet_tags=["dirm_u0"], conformity=:H1)
Vφ = TestFESpace(model, reffeφ, labels=labels, dirichlet_tags=["dire_mid", "dire_top"], conformity=:H1)
V  = MultiFieldFESpace([Vu, Vφ])



#Setup integration
degree = order+1
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)
  

# # Weak form
function res((u, φ), (v, vφ))
  return ∫((∇(v) ⊙ (∂Ψu ∘ (∇(u), ∇(φ)))) + (∇(vφ) ⋅ (∂Ψφ ∘ (∇(u), ∇(φ))))) * dΩ
end
  
 
function jac((u, φ), (du, dφ) , (v, vφ))
  return ∫(∇(v)  ⊙ (inner42∘((∂Ψuu ∘ (∇(u),∇(φ))), ∇(du)))) * dΩ +
   ∫(∇(dφ) ⋅ (inner32∘((∂Ψφu ∘ (∇(u),∇(φ))), ∇(v))))  * dΩ  +
    ∫(∇(vφ) ⋅ (inner32∘((∂Ψφu ∘ (∇(u),∇(φ))), ∇(du)))) * dΩ + 
    ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (∇(u),∇(φ))) ⋅ ∇(dφ))) * dΩ 
end

# Setup non-linear solver
nls = NLSolver(
    show_trace=true,
    method=:newton)

solver = FESolver(nls)



function run(x0, φap, step, nsteps, cache)

  #Define trial FESpaces from Dirichlet values
  u0 = VectorValue(0.0, 0.0,0.0)
  φ_bot = 0.0
  Uu = TrialFESpace(Vu, [u0])
  Uφ = TrialFESpace(Vφ, [φ_bot, φap])
  U = MultiFieldFESpace([Uu, Uφ])

  #FE problem
  op = FEOperator(res,jac, U, V)

  println("\n+++ Solving for φap $φap in step $step of $nsteps +++\n")
  ph = FEFunction(U, x0)

  ph, cache = solve!(ph, solver, op, cache)

   writevtk(Ωₕ, "results/results_$(lpad(step,3,'0'))", cellfields=["uh" => ph[1], "phi" => ph[2]])

  return get_free_dof_values(ph), cache
end
#end

function runs()
  φ_max = 0.01
  φ_inc = 0.001
  nsteps = ceil(Int, abs(φ_max) / φ_inc)

  x0 = zeros(Float64, num_free_dofs(V))

  cache = nothing
  for step in 1:nsteps
     φap = step * φ_max / nsteps
     x0, cache = run(x0, φap, step, nsteps, cache)
  end

end

@time runs()
