using Gridap
using GridapGmsh
using Gridap.TensorValues
using ForwardDiff
using BenchmarkTools
using LinearAlgebra
using Mimosa
using NLopt

# Initialisation result folder
mesh_file = "./models/mesh_platebeam_elec.msh"
result_folder = "./results/ex6"
setupfolder(result_folder)

# Material parameters
const λ = 10.0
const μ = 1.0
const ε = 1.0
const autodif = true

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

# model
#include("electro/model_electrobeam.jl")
model = GmshDiscreteModel(mesh_file)
writevtk(model, result_folder)

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
ndofm::Int =  num_free_dofs(Vu)
ndofe::Int =  num_free_dofs(Vφ)

#Setup integration
degree =  2 * order
Ωₕ     =  Triangulation(model)
dΩ     =  Measure(Ωₕ, degree)

#tuple for fem parameters
Uφ_volume      =  FESpace(model, reffeφ, conformity=:H1)
Γ              =  BoundaryTriangulation(model, tags = "dire_top")
Uφ_topSurface  =  FESpace(Γ, reffeφ)

fem_params  =  (; Ωₕ, dΩ, ndofm, ndofe, Uφ_volume, Uφ_topSurface)



# Setup non-linear solver
nls = NLSolver(
  show_trace=false,
  method=:newton,
  iterations=20)

solver = FESolver(nls)

#---------------------------------------------
#---------------------------------------------
# Tools for optimization
#---------------------------------------------
#---------------------------------------------
#utarget(x)=-((0.1*40.0)*(x[3]/40.0)^2.0)  
utarget(x)=VectorValue([0.0, -((0.3*40.0)*(x[3]/40.0)^2.0),  0.0])
N    =  VectorValue(0.0,1.0,0.0)

function ObjectiveFunctionIntegrand(uh_, Nh_, target)
#  return 0.5*(uh_⋅Nh_-target)*(uh_⋅Nh_- target)  
  return 0.5*((uh_-target)⋅N)*((uh_- target)⋅N)  
end

function VectorStateEquation(u,φ,v,vφ)
  return (∇(v)' ⊙ (∂Ψu ∘ (∇(u)', ∇(φ)))) + (∇(vφ)' ⋅ (∂Ψφ ∘ (∇(u)', ∇(φ))))
end

function JacobianStateEquation(u,φ,du, dφ, v,vφ)
  return ∇(v)' ⊙ (inner42 ∘ ((∂Ψuu ∘ (∇(u)', ∇(φ))), ∇(du)')) +
         ∇(dφ) ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(u)', ∇(φ))), ∇(v)'))  +
         ∇(vφ)' ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(u)', ∇(φ))), ∇(du)')) +
         ∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ))) ⋅ ∇(dφ))
end

function MatrixElectroProblem(φ,vφ,uh_)
   return ∇(vφ) ⋅ (∂Ψφ ∘ (∇(uh_), ∇(φ)))
end

function VectorElectroProblem(vφ)
  return vφ*0.0
end

function ResidualAdjointProblem(v, vφ, uh_, Nh_,params)
  return (((uh_ - params.utarget)⋅Nh_)*(Nh_⋅v) + vφ*0.0)
end 



function IntegrandDescendDirection(uh_,φh_, puh_, pφh_, vφ)
  return -∇(vφ) ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(uh_)', ∇(φh_))), ∇(puh_)'))  -
         ∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(uh_)', ∇(φh_))) ⋅ ∇(pφh_))
end




function NewtonRaphson(x0, φap, φ_max, loadinc, ndofm, cache)
  #----------------------------------------------
  #Define trial FESpaces from Dirichlet values
  #----------------------------------------------
  u0                   =  VectorValue(0.0, 0.0, 0.0)
  φ_mid                =  0.0
  Uu                   =  TrialFESpace(Vu, [u0])
  Uφ                   =  TrialFESpace(Vφ, [φ_mid, φap])
  U                    =  MultiFieldFESpace([Uu, Uφ])
  #----------------------------------------------
  #Update Dirichlet values solving electro problem
  #----------------------------------------------
  x0_old               =  copy(x0)
  uh                   =  FEFunction(Uu, x0[1:ndofm])
  MatrixElectro(φ,vφ)  =  ∫(MatrixElectroProblem(φ,vφ,uh))*dΩ
  VectorElectro(vφ)    =  ∫(VectorElectroProblem(vφ))*dΩ
  opφ                  =  AffineFEOperator(MatrixElectro, VectorElectro, Uφ, Vφ)
  φh                   =  solve(opφ)
  x0[ndofm+1:end]      =  get_free_dof_values(φh)
  ph                   =  FEFunction(U, x0)
  #----------------------------------------------
  #Coupled FE problem
  #----------------------------------------------
  VectorCoupled((u,φ),(v,vφ))          =  ∫(VectorStateEquation(u,φ,v,vφ))*dΩ
  MatrixCoupled((u,φ),(du,dφ),(v,vφ))  =  ∫(JacobianStateEquation(u,φ,du, dφ, v,vφ))*dΩ
  op                                   =  FEOperator(VectorCoupled, MatrixCoupled, U, V)
  loadfact                             =  round(φap / φ_max, digits=2)
  #println("\n+++ Loadinc $loadinc:  φap $φap in loadfact $loadfact +++\n")
  cacheold                             =  cache
  ph, cache                            =  solve!(ph, solver, op, cache)
  flag::Bool                           =  (cache.result.f_converged || cache.result.x_converged)
  #----------------------------------------------
  #Check convergence
  #----------------------------------------------
  if (flag == true)
    writevtk(Ωₕ, "results/ex6/results_$(loadinc)", cellfields=["uh" => ph[1], "phi" => ph[2]])
    return get_free_dof_values(ph), cache, flag
  else
    return x0_old, cacheold, flag
  end
end

function StateEquation(φ_max, fem_params)
  nsteps     =  30
  φ_inc      =  φ_max / nsteps
  x0         =  zeros(Float64, num_free_dofs(V))
  cache      =  nothing
  φap        =  0.0
  loadinc    =  0
  maxbisect  =  10
  nbisect    =  0
  setupfolder("results/ex2")
  while (φap / φ_max) < 1.0 - 1e-6
    φap   += φ_inc
    φap    = min(φap, φ_max)
    x0, cache, flag = NewtonRaphson(x0, φap, φ_max, loadinc, fem_params.ndofm, cache)
    if (flag == false)
      φap     -= φ_inc
      φ_inc    = φ_inc / 2
      nbisect += 1
    end
    if nbisect > maxbisect
      println("Maximum number of bisections reached")
      break
    end
    loadinc += 1
  end
  return x0
end


#---------------------------------------------
#---------------------------------------------
# Adjoint equation
#---------------------------------------------
#---------------------------------------------
function AdjointEquation(φ_max, xState; fem_params, obj_params)
  #----------------------------------------------
  #Define trial FESpaces from Dirichlet values
  #----------------------------------------------
  u                      =  xState[1:fem_params.ndofm];
  φ                      =  xState[fem_params.ndofm+1:end];
  
  u0                     =  VectorValue(0.0, 0.0, 0.0)
  φ_mid                  =  0.0
  Uu                     =  TrialFESpace(Vu, [u0])
  Uφ                     =  TrialFESpace(Vφ, [φ_mid, φ_max])
  uh                     =  FEFunction(Uu,u)
  φh                     =  FEFunction(Uφ,φ)
  Nh                     =  interpolate_everywhere(obj_params.N,Uu)
  #----------------------------------------------
  #Vector and Stiffness for adjoint problem
  #----------------------------------------------
  Vector_((v,vφ))         =  ∫(ResidualAdjointProblem(v, vφ, uh, Nh, obj_params))*dΩ
  Matrix_((p,pφ),(v,vφ))  =  ∫(JacobianStateEquation(uh,φh,p, pφ, v, vφ))*dΩ
  op                     =  AffineFEOperator(Matrix_, Vector_, V, V)
  ph                     =  solve(op)
  return get_free_dof_values(ph)
end

#---------------------------------------------
#---------------------------------------------
# Objective Function
#---------------------------------------------
#---------------------------------------------
function ObjectiveFunction(xState; fem_params, obj_params)
  u    =  xState[1:fem_params.ndofm];
  u0   =  VectorValue(0.0, 0.0,0.0)
  Uu   =  TrialFESpace(Vu, [u0])
  uh   =  FEFunction(Uu,u)
  Nh   =  interpolate_everywhere(obj_params.N,Uu)
  Qₕ   =  CellQuadrature(fem_params.Ωₕ,4*2)  
  return ∑(∫(ObjectiveFunctionIntegrand(uh, Nh, obj_params.utarget))Qₕ)
end

#---------------------------------------------
#---------------------------------------------
# Objective Function
#---------------------------------------------
#---------------------------------------------
function DescendDirection(xState, AdjointState; fem_params, obj_params)
  #----------------------------------------------
  #Define trial FESpaces from Dirichlet values
  #----------------------------------------------
  u              =  xState[1:fem_params.ndofm];
  φ              =  xState[fem_params.ndofm+1:end];
  pu             =  AdjointState[1:fem_params.ndofm];
  pφ             =  AdjointState[fem_params.ndofm+1:end];
  
  u0             =  VectorValue(0.0, 0.0, 0.0)
  φ_mid          =  0.0

  Uu             =  TrialFESpace(Vu, [u0])
  Uφ             =  TrialFESpace(Vφ, [φ_mid, obj_params.φmax])
  uh             =  FEFunction(Uu,u)
  puh            =  FEFunction(Vu,pu)
  φh             =  FEFunction(Uφ,φ)
  pφh            =  FEFunction(Vφ,pφ)
  #----------------------------------------------
  #Vector of sensitivities in the volume
  #----------------------------------------------
  Vector_(vφ)     =  ∫(IntegrandDescendDirection(uh,φh, puh, pφh, vφ))*dΩ
  DL_Dφmax_vol   =  assemble_vector(Vector_, fem_params.Uφ_volume)

  DL_Dφmaxh_vol  =  FEFunction(fem_params.Uφ_volume, DL_Dφmax_vol)
  DL_Dφmaxh_surf =  interpolate_everywhere(DL_Dφmaxh_vol, fem_params.Uφ_topSurface)
  DL_Dφmax_surf  =  get_free_dof_values(DL_Dφmaxh_surf)

  DL_Dφmax       =  [sum(DL_Dφmax_surf)]

  return DL_Dφmax
end

#---------------------------------------------
# Initialization of optimization variables
#---------------------------------------------
φmax   =  0.2
xini   =  [0.01]
grad   =  [0.0]
obj_params = (; N, utarget, φmax)



function f(x::Vector,grad::Vector; fem_params, obj_params)
  φ_app         =  x[1]*obj_params.φmax
  xState        =  StateEquation(φ_app, fem_params)
  AdjointState  =  AdjointEquation(φ_app, xState; fem_params, obj_params)    
  if length(grad)>0
     dobjdΦ     =  DescendDirection(xState, AdjointState; fem_params, obj_params)
     grad[:]    =  obj_params.φmax*dobjdΦ
  end
  Obj           =  ObjectiveFunction(xState; fem_params, obj_params)
  println("\n-------------------------\n")
  @show(x[1])
  @show(Obj)
  @show(φ_app)
  return  Obj
end


function electro_optimize(x_init; TOL=1e-4, MAX_ITER=500, fem_params, obj_params)
  ##################### Optimize #################
  opt                 =  Opt(:LD_MMA, 1)
  opt.lower_bounds    =  0
  opt.upper_bounds    =  1
  opt.ftol_rel        =  TOL
  opt.maxeval         =  MAX_ITER
  opt.min_objective   =  (x0, grad) -> f(x0,grad; fem_params, obj_params)

  (f_opt, x_opt, ret) =  optimize(opt, x_init)
  @show numevals      =  opt.numevals # the number of function evaluations
  return f_opt, x_opt, ret
end



#φ_app         =  0.1*obj_params.φmax
#xState        =  StateEquation(φ_app, fem_params)
#AdjointState  =  AdjointEquation(φ_app, xState; fem_params, obj_params)    
#dobjdΦ     =  DescendDirection(xState, AdjointState; fem_params, obj_params)
#Obj           =  ObjectiveFunction(xState; fem_params, obj_params)

@time f(xini,grad ; fem_params, obj_params)

# #@show f([0.47],grad::Vector; fem_params, obj_params)
# f_opt, x_opt, ret  =  electro_optimize(xini; TOL=1e-8, MAX_ITER=500, fem_params, obj_params)


# φ_app          =  x_opt[1]*obj_params.φmax
# ##φ_app          =  0.47*obj_params.φmax
# xState         =  StateEquation(φ_app, fem_params)
# u              =  xState[1:fem_params.ndofm];
# u0             =  VectorValue(0.0, 0.0, 0.0)
# Uu             =  TrialFESpace(Vu, [u0])
# uh             =  FEFunction(Uu,u)
# uTargeth       =  interpolate_everywhere(obj_params.utarget,Uu)

# writevtk(Ωₕ, "results/OptimalSolution", cellfields=["uh" => uh])
# writevtk(Ωₕ, "results/Target", cellfields=["uh" => uTargeth])


