using Gridap
using GridapGmsh
using Gridap.TensorValues
import Base: *
using LineSearches: BackTracking
using ForwardDiff
using BenchmarkTools
using LinearAlgebra
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
E(∇φ)=-∇φ
EE(∇φ) =  E(∇φ) ⋅ E(∇φ)
Ψ(∇φ) = (-ε/2 ) * EE(∇φ)

∂Ψ_∂∇φ(∇φ)       =  ForwardDiff.gradient(Ψ, get_array(∇φ))
∂2Ψ_∂2∇φ(∇φ)     =  ForwardDiff.hessian(Ψ, get_array(∇φ))

∂Ψφ(∇φ)       = VectorValue(∂Ψ_∂∇φ(∇φ))
∂Ψφφ(∇φ)      = TensorValue(∂2Ψ_∂2∇φ( ∇φ))

 

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
order = 1
reffeφ = ReferenceFE(lagrangian, Float64, order)

#Define test FESpaces
V = TestFESpace(model, reffeφ, labels=labels, dirichlet_tags=["dire_mid", "dire_top"], conformity=:H1)



#Setup integration
degree = order+1
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)
  

# # Weak form
function res(φ, vφ)
  return ∫(∇(vφ) ⋅ (∂Ψφ ∘ (∇(φ)))) * dΩ
end
  
 
function jac(φ, dφ, vφ)
  return ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (∇(φ))) ⋅ ∇(dφ))) * dΩ 
end

# Setup non-linear solver
nls = NLSolver(
    show_trace=true,
    method=:newton)

solver = FESolver(nls)



function run(x0, φap, step, nsteps, cache)

  #Define trial FESpaces from Dirichlet values
  φ_bot = 0.0
  U = TrialFESpace(V, [φ_bot, φap])

  #FE problem
  op = FEOperator(res, jac, U, V)

  println("\n+++ Solving for φap $φap in step $step of $nsteps +++\n")
  ph = FEFunction(U, x0)

  ph, cache = solve!(ph, solver, op, cache)


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
