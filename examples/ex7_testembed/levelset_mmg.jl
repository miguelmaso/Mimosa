
using Gridap
using GridapEmbedded
using LinearAlgebra: tr
using Mimosa
using Gridap.Geometry
using Gridap.Adaptivity
using GridapGmsh
using GridapEmbedded.LevelSetCutters

# Initialisation result folder
result_folder = "./results/ex7"
setupfolder(result_folder)
 
 
function optdomain(x)
    center = Point(1.0,0.,0.)
    r=0.7
    distancia = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2 +
     (x[3] - center[3])^2) - r
    # Retornar el resultado
    return distancia
end
 

function sph(x)
    center = Point(0.,0.,0.)
    r=0.7
    distancia = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2 +
     (x[3] - center[3])^2) - r
    # Retornar el resultado
    return distancia
end
 

 cartesian=true
 
if cartesian==true
 n=20
 L = VectorValue(1,1,1)
 partition = (L[1]*n,L[2]*n,L[3]*n)
 pmin = Point(0.,0.,0.)
 pmax = pmin + L
 bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
  # Identify Dirichlet boundaries
  labeling = get_face_labeling(bgmodel)
  add_tag_from_tags!(labeling,"support0",[1,3,5,7,13,15,17,19,25])
  add_tag_from_tags!(labeling,"support1",[2,4,6,8,14,16,18,20,26])
  else
 mesh_file = "./models/gmsh_cube.msh"
 bgmodel = GmshDiscreteModel(mesh_file)
end


writevtk(bgmodel, "results/ex7/model") 

# Level-set function
fes::FESpace= FESpace(bgmodel, ReferenceFE(lagrangian, Float64, 1),
 vector_type=Vector{Float64}, conformity=:H1)
Ω_bg  = Triangulation(bgmodel)
 

# #----------------------------------------------
# # Level-set function OPTION 1
# #----------------------------------------------
geo1=AnalyticalGeometry(sph)
geo1_d = discretize(geo1,bgmodel) 
lvl_set = geo1_d.tree.data[1]


# #----------------------------------------------
# # Level-set function OPTION 2
# #----------------------------------------------
ψₕ = interpolate_everywhere(sph, fes)
ψ =get_free_dof_values(ψₕ)
writevtk(Ω_bg, "results/ex7/ψₕ", cellfields=["ψₕ" => ψₕ])
ψgeo=AnalyticalGeometry(x->ψₕ(x))

geo2=AnalyticalGeometry(optdomain)
fullgeo = intersect(geo1,geo2)
# fullgeo_d = discretize(fullgeo,bgmodel) 
# lvl_set2 = fullgeo_d.tree.data[1]
# lvl_seth=FEFunction(fes,lvl_set2)
# writevtk(Ω_bg, "results/ex7/ψₕ", cellfields=["ψₕ" => ψₕ])

cutgeo = cut(bgmodel,fullgeo)
 
 Ω1_act = Triangulation(cutgeo,ACTIVE)
 Ω1_phys = Triangulation(cutgeo,PHYSICAL)

writevtk(Ω1_act, "results/ex7/Ω1_act" )
 writevtk(Ω1_phys, "results/ex7/Ω1_phys" )
 
 using Gridap.ReferenceFEs

