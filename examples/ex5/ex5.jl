using ForwardDiff: ceil
using Gridap
using GridapGmsh
using Gridap.TensorValues
using Gridap.Algebra
using LineSearches: BackTracking
using ForwardDiff
using BenchmarkTools
using LinearAlgebra
using Mimosa
using StaticArrays
using NLopt


setupfolder("results/ex5") 

Ba   =  VectorValue(0.0, 0.0, 15e-5)

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
∂2Ψ_∂2∇uBr(∇u,Br,Ba)   =  ForwardDiff.jacobian(Br -> ∂Ψ_∂∇u(∇u, get_array(Br), get_array(Ba)), get_array(Br))



∂Ψu(∇u,Br,Ba)       = TensorValue(∂Ψ_∂∇u(∇u,Br,Ba))
∂Ψuu(∇u,Br,Ba)      = TensorValue(∂2Ψ_∂2∇u(∇u,Br,Ba))
∂ΨuBr(∇u,Br,Ba)     = TensorValue(∂2Ψ_∂2∇uBr(∇u,Br,Ba)) 


# model
mesh_file = "./models/mesh_platebeam_mag.msh"
model = GmshDiscreteModel(mesh_file) 
labels = get_face_labeling(model)
 add_tag_from_tags!(labels, "dirm_u0",  [1])
writevtk(model, "results/model")

 
#Define reference FE (Q2/P1(disc) pair)
order = 1
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

#Define test FESpaces
V = TestFESpace(model, reffe, labels=labels, dirichlet_tags=["dirm_u0"], conformity=:H1)


#Setup integration
degree = 2*order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)



nelem      =  num_cells(Ωₕ)
hmin   =  minimum(get_cell_measure(Ωₕ))

#Definie optimization finite element spaces
angleL2_fe    =  ReferenceFE(lagrangian, Float64, 0)
angleL2_fes   =  FESpace(Ωₕ, angleL2_fe, vector_type  =  Vector{Float64}, conformity=:L2)

angleH1_fe    =  ReferenceFE(lagrangian, Float64, 1)
angleH1_fes   =  FESpace(Ωₕ, angleH1_fe, vector_type  =  Vector{Float64}, conformity=:H1)

BrH1_fe       =  ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
BrH1_fes      =  FESpace(Ωₕ, BrH1_fe, vector_type  =  Vector{Float64}, conformity=:H1)



r           =  1.3*hmin
angleL2_fes_  =  angleL2_fes
npoints::Int = num_free_dofs(BrH1_fes)/3
fem_params  =  (; nelem, npoints, angleL2_fes, BrH1_fes, angleH1_fes, dΩ , Ωₕ, r, Ba)
 
#---------------------------------------------
# Setup non-linear solver in State and Adjoint equations
#---------------------------------------------
nls = NLSolver(LUSolver(),
     show_trace=true,
     method=:newton)
solver = FESolver(nls)
 

#---------------------------------------------
#---------------------------------------------
# Tools for optimization
#---------------------------------------------
#---------------------------------------------
utarget(x)=((0.1*40.0)*(x[3]/40.0)^2.0)  
N    =  VectorValue(0.0,0.0,1.0)
Obj_Params = (; N, utarget)

function ObjectiveFunctionIntegrand(uh_, Nh_, target)
  return 0.5*(uh_⋅Nh_-target)*(uh_⋅Nh_- target)  
end

function ResidualStateEquation(u, v, Brh_, Bah_)
  return ((∇(v)' ⊙ (∂Ψu ∘ (∇(u)', Brh_, Bah_)))) 
end 

function JacobianStateEquation(u, du , v, Brh_, Bah_)
return (∇(v)'  ⊙ (inner42∘((∂Ψuu ∘ (∇(u)',Brh_, Bah_)), ∇(du)'))) 
end 

function ResidualAdjointProblem(v, uh_, Nh_,params)
  return ((uh_⋅Nh_-params.utarget)*(Nh_⋅v))
end 

function DescendDirectionIntegrand(v,ph_, uh_, Brh_, Bah_, ∂Br_∂xh)
  return (-(∇(ph_)'  ⊙ (inner31∘((∂ΨuBr ∘ (∇(uh_)',Brh_, Bah_)), ∂Br_∂xh)))*v)
end 


a_f(r, u, v) = r^2*(∇(v) ⋅ ∇(u))
function FilterMatrix(r, u,v)
  return  a_f(r, u, v) + v*u
end
function FilterResidual(v, p0)
  return  (v * p0)
end


function Filter(p0; fem_params)
  pfh            = FEFunction(fem_params.angleL2_fes, p0)
  Matrix(pfh,v)  =  ∫(FilterMatrix(fem_params.r,pfh,v))*fem_params.dΩ
  Vector(v)      =  ∫(FilterResidual(v,p0))*fem_params.dΩ
  op             =  AffineFEOperator(Matrix,Vector,fem_params.angleH1_fes, fem_params.angleH1_fes)
  pfh            = solve(op)  
  return get_free_dof_values(pfh)
end


#---------------------------------------------
# Mapping from angles (L2) to Br (H1) 
#---------------------------------------------
function mapBr(x::Vector{Float64}, fem_params)
  #-----------------------------------
  # Get θ and φ at elements from x
  #-----------------------------------
  θ      =  x[1:fem_params.nelem]
  φ      =  x[fem_params.nelem+1: 2*fem_params.nelem]
  θH1    =  Filter(θ; fem_params)
  φH1    =  Filter(φ; fem_params)
  f(x,y) =  [cos(x)*sin(y), sin(x)*sin(y), cos(y)]
  return reduce(vcat, f.(θH1,φH1))::Vector{Float64}
end

function map∂Br∂θ(x::Vector{Float64}, fem_params)
  #-----------------------------------
  # Get θ and φ at elements from x
  #-----------------------------------
  θ      =  x[1:fem_params.nelem]
  φ      =  x[fem_params.nelem+1: 2*fem_params.nelem]
  θH1    =  Filter(θ; fem_params)
  φH1    =  Filter(φ; fem_params)
  f(x,y) =  [-sin(x)*sin(y), cos(x)*sin(y), 0.0]
  return reduce(vcat, f.(θH1,φH1))
end


function map∂Br∂φ(x::Vector{Float64}, fem_params)
  #-----------------------------------
  # Get θ and φ at elements from x
  #-----------------------------------
  θ      =  x[1:fem_params.nelem]
  φ      =  x[fem_params.nelem+1: 2*fem_params.nelem]
  θH1    =  Filter(θ; fem_params)
  φH1    =  Filter(φ; fem_params)
  f(x,y) =  [cos(x)*cos(y), sin(x)*cos(y), -sin(y)]
  return reduce(vcat, f.(θH1,φH1))
end


#---------------------------------------------
#---------------------------------------------
# State equation
#---------------------------------------------
#---------------------------------------------
function StateEquationIter(u, Bah, Brh, step, nsteps, cache)
  # # Weak form
  Vector(u,v)  =  ∫(ResidualStateEquation(u, v, Brh, Bah))*dΩ
  Matrix(u,du,v)  =  ∫(JacobianStateEquation(u, du , v, Brh, Bah))*dΩ
  #Define trial FESpaces from Dirichlet values
  u0         =  VectorValue(0.0, 0.0,0.0)
  U          =  TrialFESpace(V, [u0])
  uh         =  FEFunction(U, u)
  #Update Dirichlet values FE problem
  op          =  FEOperator(Vector,Matrix,U,V)
  uh, cache  =  solve!(uh, solver, op, cache)
  return get_free_dof_values(uh), cache
end

function StateEquation(Φ; fem_params)

  Br           =  mapBr(Φ, fem_params)  
  Brh          =  FEFunction(fem_params.BrH1_fes,Br)
  Bapp_inc     =  1.0/40.0
  nsteps       =  ceil(Int, 1 / Bapp_inc)
  u            =  zeros(Float64, num_free_dofs(V))
  cache        =  nothing
  for step in 1:nsteps
    Λ          =  step / nsteps
    Ba_app     =  fem_params.Ba*Λ
    Bah        =  interpolate_everywhere(Ba_app,V)  
    u, cache   =  StateEquationIter(u, Bah, Brh, step, nsteps, cache)
  end
  return u
end
#---------------------------------------------
#---------------------------------------------
# Adjoint equation
#---------------------------------------------
#---------------------------------------------
function AdjointEquation(u , Φ; fem_params, Obj_Params)

  Br           =  mapBr(Φ, fem_params)  
  Brh          =  FEFunction(fem_params.BrH1_fes,Br)
  Bah          =  interpolate_everywhere(fem_params.Ba,V)  
  u0           =  VectorValue(0.0, 0.0,0.0)
  U            =  TrialFESpace(V, [u0])
  uh           =  FEFunction(U,u)
  p            =  zeros(Float64, num_free_dofs(V))
  ph           =  FEFunction(V,p)
  Nh           =  interpolate_everywhere(Obj_Params.N,U)

  Vector(v)    =  ∫(ResidualAdjointProblem(v, uh, Nh,Obj_Params))*dΩ
  Matrix(p,v)  =  ∫(JacobianStateEquation(uh, p, v, Brh, Bah))*dΩ
  op           =  AffineFEOperator(Matrix, Vector, V, V)
  ph           =  solve(op)
  return get_free_dof_values(ph)
end


#---------------------------------------------
#---------------------------------------------
# Objective Function
#---------------------------------------------
#---------------------------------------------
function ObjectiveFunction(u; fem_params, Obj_Params)
  u0   =  VectorValue(0.0, 0.0,0.0)
  U    =  TrialFESpace(V, [u0])
  uh   = FEFunction(U,u)
  Nh   =  interpolate_everywhere(Obj_Params.N,U)
  Qₕ   = CellQuadrature(fem_params.Ωₕ,4*2)  
  return ∑(∫(ObjectiveFunctionIntegrand(uh, Nh, Obj_Params.utarget))Qₕ)
end


#---------------------------------------------
#---------------------------------------------
# Derivative of {θf,φf}∈H¹(Ω) with respect
# to {θ,φ}∈L²(Ω)
#---------------------------------------------
#---------------------------------------------
function DescendDirection(Φ, u, p; fem_params)

  Br           =  mapBr(Φ, fem_params)  
  ∂Br_∂θ       =  map∂Br∂θ(Φ, fem_params)
  ∂Br_∂φ       =  map∂Br∂φ(Φ, fem_params)

  Brh          =  FEFunction(fem_params.BrH1_fes,Br)
  ∂Br_∂θh      =  FEFunction(fem_params.BrH1_fes,∂Br_∂θ)
  ∂Br_∂φh      =  FEFunction(fem_params.BrH1_fes,∂Br_∂φ)

  Bah          =  interpolate_everywhere(Ba,V)  
  u0           =  VectorValue(0.0, 0.0,0.0)
  U            =  TrialFESpace(V, [u0])
  uh           =  FEFunction(U,u)
  ph           =  FEFunction(V,p)

  Matrix(w,v)  =  ∫(FilterMatrix(fem_params.r,w,v))*fem_params.dΩ
  Vectorθ(v)    =  ∫(DescendDirectionIntegrand(v,ph, uh, Brh, Bah, ∂Br_∂θh))*dΩ
  Vectorφ(v)    =  ∫(DescendDirectionIntegrand(v,ph, uh, Brh, Bah, ∂Br_∂φh))*dΩ

  opθ         =  AffineFEOperator(Matrix, Vectorθ, fem_params.angleH1_fes, fem_params.angleH1_fes)
  ∂L_∂θfh     =  solve(opθ)
  opφ         =  AffineFEOperator(Matrix, Vectorφ, fem_params.angleH1_fes, fem_params.angleH1_fes)
  ∂L_∂φfh     =  solve(opφ)
  DL_Dθ(v)    =  ∫(∂L_∂θfh*v)*dΩ
  DL_Dφ(v)    =  ∫(∂L_∂φfh*v)*dΩ
  ∂L_∂θ       =  assemble_vector(DL_Dθ,fem_params.angleL2_fes)
  ∂L_∂φ       =  assemble_vector(DL_Dφ,fem_params.angleL2_fes)
  return      [∂L_∂θ;∂L_∂φ]  
end
 

#---------------------------------------------
# Initialization of optimization variables
#---------------------------------------------


function f(x::Vector, grad::Vector; fem_params, Obj_Params)
  Φ=map(p->(2.0*pi)*(2.0*p-1.0), x)  
  u    =  StateEquation(Φ;fem_params)
  p    =  AdjointEquation(u, Φ; fem_params, Obj_Params)
  if length(grad) > 0
    dobjdΦ =  DescendDirection(Φ, u, p; fem_params)
    grad[:] = 4.0*pi*dobjdΦ
  end
  @show fo=ObjectiveFunction(u; fem_params, Obj_Params)
  return fo
end


θini=0.5
φini=0.5

xθ = fill((θini/(2.0*pi) +1.0)/2.0, nelem)
xφ = fill((φini/(2.0*pi) +1.0)/2.0, nelem)
x0 = [xθ;xφ]
grad = zeros(2*nelem)

function magnet_optimize(x_init; TOL = 1e-4, MAX_ITER = 500, fem_params, Obj_Params)
  ##################### Optimize #################
  opt = Opt(:LD_MMA, length(x_init))
  opt.lower_bounds = 0
  opt.upper_bounds = 1
  opt.ftol_rel = TOL
  opt.maxeval = MAX_ITER
  opt.min_objective =   (x0, grad) -> f(x0, grad; fem_params, Obj_Params)
  (f_opt, x_opt, ret) = optimize(opt, x_init)
  @show numevals = opt.numevals # the number of function evaluations
  return f_opt, x_opt, ret
end



a, b, ret=magnet_optimize(x0; TOL = 1e-8, MAX_ITER=500, fem_params, Obj_Params)
@show ret
# @show(norm(u))
# @show(norm(p))
# @show(f)
# @show(norm(a))

#writevtk(Ωₕ, "results/u3vis", cellfields=["u3h" => u3h])

