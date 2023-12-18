using Gridap
using Gridap.TensorValues
using Gridap.Geometry
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Arrays
using Gridap.ReferenceFEs


function box(p ; L=[1.0,1.0,1.0], Centroid=Point(0.,0.,0.))
 z1=Centroid[3]-L[3]*0.5
 z2=Centroid[3]+L[3]*0.5
 x1=Centroid[2]-L[1]*0.5
 x2=Centroid[2]+L[1]*0.5
 y1=Centroid[1]-L[2]*0.5
 y2=Centroid[1]+L[2]*0.5
     return -min(min(min(min(min(-z1+p[3],z2-p[3]),-y1+p[2]),y2-p[2]),-x1+p[1]),x2-p[1])
end
 
 
 



function optdomain(x)
    center = Point(1.0,0.,0.)
    r=1.0
    distancia = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2 +
     (x[3] - center[3])^2) - r
    # Retornar el resultado
    return distancia
end

 

n=2
L = VectorValue(1,1,1)
partition = (L[1]*n,L[2]*n,L[3]*n)
pmin = Point(0.,0.,0.)
pmax = pmin + L
bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
writevtk(bgmodel, "results/ex7/model") 

 # Identify Dirichlet boundaries
 labeling = get_face_labeling(bgmodel)
#  add_tag_from_tags!(labeling,"support0",[1,3,5,7,13,15,17,19,25])
#  add_tag_from_tags!(labeling,"support1",[2,4,6,8,14,16,18,20,26])

@show get_tag_from_name(labeling)
@show labeling.tag_to_entities
@show labeling.tag_to_name
@show labeling.d_to_dface_to_entity[2]



geo=AnalyticalGeometry(x->box(x; L=[0.5,0.5,0.5],Centroid=Point(0.,0.,0.) ))
geo1_d = discretize(geo,bgmodel) 
nodesfinded= findall(x -> x < 0, geo1_d.tree.data[1])
 
fullgeo=AnalyticalGeometry(optdomain)
cutgeo = cut(bgmodel,fullgeo)
Γd = EmbeddedBoundary(cutgeo)
SubFacetData(Γd)




labels.d_to_dface_to_entity[1][[Γ_N_bt_N_nums;Γ_N_tp_N_nums]] .= entity
        labels.d_to_dface_to_entity[2][[Γ_N_bt_L_nums;Γ_N_tp_L_nums]] .= entity
        add_tag!(labels,"Gamma_D",[entity])


 point_to_coords = collect1d(get_node_coordinates(bgmodel))
 
 @show aa=compute_face_nodes(bgmodel,1)
 @show aa=get_face_mask(labeling,2)

 

# map(optdomain, bgmodel.grid.node_coords)
 
# geo(bgmodel.grid.node_coords[1])

# function update_labels!(model,elx,ely)
#     cell_to_entity = map_parts(local_views(model)) do model
#         labels = get_face_labeling(model)
#         cell_to_entity = labels.d_to_dface_to_entity[end]
#         entity = maximum(cell_to_entity) + 1
#         prop_Γ_N = 0.4
#         prop_Γ_D = 0.2
#         Γ_N_mx_el = ceil(Int,prop_Γ_N*ely)
#         Γ_N_mx_nl = ceil(Int,prop_Γ_N*(ely+1))
#         Γ_D_mx_el = ceil(Int,prop_Γ_D*ely)
#         Γ_D_mx_nl = ceil(Int,prop_Γ_D*(ely+1))
#         # Γ_D
#         Γ_N_bt_N_nums = 1:(elx+1):(Γ_D_mx_nl)*(elx + 1)
#         Γ_N_bt_L_nums = [3;collect((3elx+3):(2elx + 1):((3elx+3)+(Γ_D_mx_el-2)*(2elx + 1)))]
#         Γ_N_tp_N_nums = ((ely-Γ_D_mx_nl+1)*(elx + 1)+1):(elx+1):((ely)*(elx + 1)+1)
#         Γ_N_tp_L_nums = ((3elx+3)+(ely-Γ_D_mx_el-1)*(2elx + 1)):(2elx + 1):((3elx+3)+(ely-2)*(2elx + 1))
#         labels.d_to_dface_to_entity[1][[Γ_N_bt_N_nums;Γ_N_tp_N_nums]] .= entity
#         labels.d_to_dface_to_entity[2][[Γ_N_bt_L_nums;Γ_N_tp_L_nums]] .= entity
#         add_tag!(labels,"Gamma_D",[entity])
#         # Γ_N
#         cell_to_entity = labels.d_to_dface_to_entity[end]
#         entity = maximum(cell_to_entity) + 1
#         Γ_D_middle_N_nums = (Γ_N_mx_nl)*(elx + 1):(elx+1):((ely-Γ_N_mx_nl+2)*(elx + 1))
#         Γ_D_middle_L_nums = ((3elx+1)+(Γ_N_mx_el-1)*(2(elx+1)-1)):(2(elx+1)-1):((3elx+1)+(ely-Γ_N_mx_el)*(2(elx+1)-1))
#         labels.d_to_dface_to_entity[1][Γ_D_middle_N_nums] .= entity
#         labels.d_to_dface_to_entity[2][Γ_D_middle_L_nums] .= entity
#         add_tag!(labels,"Gamma_N",[entity])
#     end
#     cell_gids=get_cell_gids(model)
#     exchange!(cell_to_entity,cell_gids.exchanger)
# end


# labels.d_to_dface_to_entity[1][[Γ_N_bt_N_nums;Γ_N_tp_N_nums]] .= entity
# labels.d_to_dface_to_entity[2][[Γ_N_bt_L_nums;Γ_N_tp_L_nums]] .= entity
# add_tag!(labels,"Gamma_D",[entity])


labeling.d_to_dface_to_entity[1][1]=33
add_tag!(labeling,"Gamma_D",[33])