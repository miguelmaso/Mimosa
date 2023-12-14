
using Gridap
using Gridap.Geometry

using IterativeSolvers: cg
using AlgebraicMultigrid
using AlgebraicMultigrid: smoothed_aggregation
using AlgebraicMultigrid: aspreconditioner

using Mimosa

solvertype = "cg"
mesh_file = "./models/crank.json"
model = DiscreteModelFromFile(mesh_file)
  
 
labels = get_face_labeling(model)
dimension = 3
tags = get_face_tag(labels,dimension)
const alu_tag = get_tag_from_name(labels,"material_1")
 
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end

const E_alu = 70.0e9
const ν_alu = 0.33
const (λ_alu,μ_alu) = lame_parameters(E_alu,ν_alu)

const E_steel = 200.0e9
const ν_steel = 0.33
const (λ_steel,μ_steel) = lame_parameters(E_steel,ν_steel)

 
function σ_bimat(ε,tag)
  if tag == alu_tag
    return λ_alu*tr(ε)*one(ε) + 2*μ_alu*ε
  else
    return λ_steel*tr(ε)*one(ε) + 2*μ_steel*ε
  end
end
 
 order = 1
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
V0 = TestFESpace(model,reffe;
  conformity=:H1,
  dirichlet_tags=["surface_1","surface_2"],
  dirichlet_masks=[(true,false,false), (true,true,true)])
 
g1(x) = VectorValue(0.005,0.0,0.0)
g2(x) = VectorValue(0.0,0.0,0.0)

U = TrialFESpace(V0,[g1,g2])
  
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

 
a(u,v) = ∫( ε(v) ⊙ (σ_bimat∘(ε(u),tags)) )*dΩ
l(v) = 0

op = AffineFEOperator(a,l,U,V0)

if solvertype == "cg"
  A = get_matrix(op)
  b = get_vector(op)
  p = aspreconditioner(smoothed_aggregation(A))
  x = cg(A,b,verbose=true,Pl=p,reltol=1e-10)
  uh = FEFunction(U,x)
 else
  uh = solve(op)
end

  setupfolder("results/ex4")
writevtk(Ω,"results/ex4/results_bimat",cellfields=
  ["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ_bimat∘(ε(uh),tags)])


 