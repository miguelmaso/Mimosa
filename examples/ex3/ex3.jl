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
mesh_file = "./models/mesh_platebeam_mag.msh"
result_folder = "./results/ex3/"
setupfolder(result_folder)

Br   =  VectorValue(0.0, 0.0, 1.0)
Ba   =  VectorValue(0.0, 15e-5, 0.0)

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
FBr_Ba(∇u,Br,Ba) =  (FBr(∇u,Br)) ⋅ Ba
Ψmec(∇u) = μ / 2 * tr((F(∇u))' * F(∇u)) - μ * logreg(J(F(∇u))) + (λ / 2) * (J(F(∇u)) - 1)^2
Ψmag(∇u,Br,Ba) = -μ0*(FBr_Ba(∇u,Br,Ba))
Ψ(∇u, Br, Ba) = Ψmec(∇u) + Ψmag(∇u,Br, Ba)

∂Ψ_∂∇u(∇u,Br,Ba)       =  ForwardDiff.gradient(∇u->Ψ(∇u,get_array(Br),get_array(Ba)), get_array(∇u))
∂2Ψ_∂2∇u(∇u,Br,Ba)     =  ForwardDiff.jacobian(∇u->∂Ψ_∂∇u(∇u,get_array(Br),get_array(Ba)), get_array(∇u))

∂Ψu(∇u,Br,Ba)       = TensorValue(∂Ψ_∂∇u(∇u,Br,Ba))
∂Ψuu(∇u,Br,Ba)      = TensorValue(∂2Ψ_∂2∇u(∇u,Br,Ba))


   # model
#include("electro/model_electrobeam.jl")

# mesh_file = joinpath(dirname(@__FILE__), "ex3_mesh.msh")
model = GmshDiscreteModel(mesh_file) 
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "dirm_u0", [1])

#Define reference FE (Q2/P1(disc) pair)
order = 1
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

#Define test FESpaces
V = TestFESpace(model, reffe, labels=labels, dirichlet_tags=["dirm_u0"], conformity=:H1)

#Setup integration
degree = 2*order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)


Brh = interpolate_everywhere(Br,V)
Bah = interpolate_everywhere(Ba,V)

# # Weak form
function res(u, v)
    return ∫((∇(v)' ⊙ (∂Ψu ∘ (∇(u)', Brh, Bah)))) * dΩ
end
 
function jac(u, du , v)
  return ∫(∇(v)'  ⊙ ((∂Ψuu ∘ (∇(u)',Brh, Bah))⊙ (∇(du)'))) * dΩ
end 
 
# # Setup non-linear solver


    nls = NLSolver(;
    show_trace=true,
    method=:newton,
    iterations=20)

solver = FESolver(nls)
pvd_results = paraview_collection(result_folder*"results", append=false)

function run(x0, Bapp, Bapp_old, Ba, step, nsteps, cache)

  Ba      =  Ba*Bapp
#   BaOld   =  Ba*Bapp_old
  println(Bapp)
  println(Bapp_old)
  @show(Ba)


#   constant_value = get_array(Ba)
  Bah            = interpolate_everywhere(Ba,V)
#   constant_value = get_array(BaOld)
#   BaOldh         = interpolate_everywhere(BaOld,V)
  uh             = FEFunction(V,x0)
  function l(v)
    return ∫(-(∇(v) ⊙ (∂Ψu ∘ (∇(uh), Brh, Bah)))) * dΩ + ∫(∇(v)  ⊙   ((∂Ψuu ∘ (∇(uh),Brh, BahOld)) ⊙ (∇(uh)))) * dΩ
    #
  end
 
  function a(u, v)
    return ∫(∇(v)  ⊙   ((∂Ψuu ∘ (∇(uh),Brh, BahOld)) ⊙ (∇(u)))) * dΩ
  end 
   
 
  #Define trial FESpaces from Dirichlet values
  u0         = VectorValue(0.0, 0.0,0.0)
  U          = TrialFESpace(V, [u0])
  
  op         = FEOperator(res,jac, U, V)
  uh, cache  = solve!(uh, solver, op, cache)
  pvd_results[step] = createvtk(Ωₕ,result_folder * "_$step.vtu", cellfields=["uh" => uh],order=2)
  # writevtk(Ωₕ, "results/results_$(lpad(step,3,'0'))", cellfields=["uh" => uh])
  return get_free_dof_values(uh), cache
end
#end


function runs()
  Bapp_max     =  1
  Bapp_inc     =  0.1
  nsteps       =  ceil(Int, abs(Bapp_max) / Bapp_inc)
  x0           =  zeros(Float64, num_free_dofs(V))
  Bapp         =  0.0
  cache        =  nothing
  for step in 1:nsteps
    Bapp_old   =  Bapp
    Bapp       =  step * Bapp_max / nsteps
    x0, cache  =  run(x0, Bapp, Bapp_old, Ba, step, nsteps, cache)
  end
  vtk_save(pvd_results)

end

@time runs()