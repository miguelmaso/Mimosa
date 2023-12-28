
using Gridap
using Gridap.FESpaces
using Mimosa

result_folder = "./results/periodic/"
setupfolder(result_folder)

nely,nelx,nelz = 21,21,21
domain = (0.0, 1, 0.0, 1, 0.0, 1);
model = CartesianDiscreteModel(domain,(nelx,nely,nelz); isperiodic=(true,true,true));

labels = get_face_labeling(model)
entity = num_entities(labels) + 1
labels.d_to_dface_to_entity[1][1] = entity
add_tag!(labels,"pt_xyz_000",[entity])

model_file = joinpath(result_folder, "model")
writevtk(model, model_file)


order = 1
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
V= TestFESpace(model, reffe, labels=labels, dirichlet_tags=["pt_xyz_000"], conformity=:H1)
u0= VectorValue(0.0,0.0,0.0)
U = TrialFESpace(V,u0)


const E = 1
const ν = 0.2
const λ = (E*ν)/((1+ν)*(1-2*ν))
const μ = E/(2*(1+ν))
σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε
 
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

f= VectorValue(0.0,0.0,0.005)

a(u,v) = ∫( ε(v) ⊙ ( σ∘ε(u)) )*dΩ
l(v) = ∫( v⋅f )*dΩ 

op = AffineFEOperator(a,l,U,V)
uh = solve(op)
 
@show ∑(∫(∇(uh))dΩ) 


writevtk(Ω,result_folder * "results",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ∘ε(uh)])
  
 