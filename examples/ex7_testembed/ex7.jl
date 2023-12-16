
using Gridap
using GridapEmbedded
using LinearAlgebra: tr
using Mimosa
using Gridap.Geometry
using Gridap.Adaptivity

# Initialisation result folder
result_folder = "./results/ex7"
setupfolder(result_folder)

function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end

 n=4
   # Background model
   L = VectorValue(3,2,3)
   partition = (L[1]*n,2*L[2]*n,2*L[3]*n)
   pmin = Point(0.,0.,0.)
   pmax = pmin + L
   bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
   bgmodelunc = UnstructuredDiscreteModel(bgmodel)
   model_file = joinpath(result_folder, "model")
   writevtk(bgmodel, model_file)
 
    model1 = refine(bgmodelunc;cells_to_refine=[1,6,16], refinement_method="red")
    Ω_bg = Triangulation(model1)

    # model1 = refine(bgmodel,4)
    model_file = joinpath(result_folder, "model1")
    writevtk(Ω_bg, model_file)

    # model2 = refine(bgmodel,Tuple(collect(1:3)))
    # model_file = joinpath(result_folder, "model2")
    # writevtk(model2, model_file)


    function sph(x)
        center = Point(0.,0.,0.)
        r=1.0
        distancia = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2 +
         (x[3] - center[3])^2) - r
        # Retornar el resultado
        return distancia
    end
    
    # Ejemplo de uso
    x =  Point(0.5,0.,0.)
    sph(x)

    fe = ReferenceFE(lagrangian, Float64, 1)
    fes = FESpace(bgmodel, fe, vector_type=Vector{Float64}, conformity=:H1)
    Ωₕ = Triangulation(bgmodel)

    lsf = interpolate_everywhere(Ωₕ, fes)
    writevtk(Ωₕ, "results/ex7/results", cellfields=["lsf" => lsf])


  
 
     
