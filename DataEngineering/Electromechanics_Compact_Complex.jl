using Pkg
Pkg.activate(".")

using Gridap
using GridapGmsh
using Gridap.TensorValues
using ForwardDiff
using Mimosa
using NLopt
using WriteVTK
using DelimitedFiles
using Base.Threads


# Initialisation result folder
#mesh_file = "../examples/ex6/parametrize_plate_elec.msh"
mesh_file = joinpath(dirname(@__FILE__), "parametrize_plate_elec_complex.msh")

result_folder = "./results/Data_Results_complex/"
setupfolder(result_folder)
# Material parameters
const λ = 10.0
const μ = 1.0
const ε = 1.0

#---------------------------
# Start of the function call
#---------------------------


function CompactCall(input_potential::Vector, folder_name)


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

    # Grid model
    model = GmshDiscreteModel(mesh_file)

    println("IM HERE")

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "fix", [31])
    add_tag_from_tags!(labels, "b1", [1])
    add_tag_from_tags!(labels, "b2", [2])
    add_tag_from_tags!(labels, "b3", [3])
    add_tag_from_tags!(labels, "b4", [4])
    add_tag_from_tags!(labels, "b5", [5])
    add_tag_from_tags!(labels, "b6", [6])
    add_tag_from_tags!(labels, "b7", [7])
    add_tag_from_tags!(labels, "b8", [8])
    add_tag_from_tags!(labels, "b9", [9])
    add_tag_from_tags!(labels, "b10", [10])
    add_tag_from_tags!(labels, "m1", [11])
    add_tag_from_tags!(labels, "m2", [12])
    add_tag_from_tags!(labels, "m3", [13])
    add_tag_from_tags!(labels, "m4", [14])
    add_tag_from_tags!(labels, "m5", [15])
    add_tag_from_tags!(labels, "m6", [16])
    add_tag_from_tags!(labels, "m7", [17])
    add_tag_from_tags!(labels, "m8", [18])
    add_tag_from_tags!(labels, "m9", [19])
    add_tag_from_tags!(labels, "m10", [20])
    add_tag_from_tags!(labels, "t1", [21])
    add_tag_from_tags!(labels, "t2", [22])
    add_tag_from_tags!(labels, "t3", [23])
    add_tag_from_tags!(labels, "t4", [24])
    add_tag_from_tags!(labels, "t5", [25])
    add_tag_from_tags!(labels, "t6", [26])
    add_tag_from_tags!(labels, "t7", [27])
    add_tag_from_tags!(labels, "t8", [28])
    add_tag_from_tags!(labels, "t9", [29])
    add_tag_from_tags!(labels, "t10", [30])
#    model_file = joinpath(result_folder, "model")
#    writevtk(model, model_file)


    #Define Finite Element Collections
    order = 2
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)

    #Setup integration
    degree = 2 * order
    Ωₕ = Triangulation(model)
    dΩ = Measure(Ωₕ, degree)

    #Define Finite Element Spaces
    Vu = TestFESpace(Ωₕ, reffeu, labels=labels, dirichlet_tags=["fix"], conformity=:H1)
    Vφ = TestFESpace(Ωₕ, reffeφ, labels=labels, dirichlet_tags=["t1", "t2","t3","t4","t5","t6","t7","t8","t9","t10","m1","m2","m3","m4","m5","m6","m7","m8","m9","m10","b1","b2","b3","b4","b5","b6","b7","b8","b9","b10"], conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    u0 = VectorValue(0.0, 0.0, 0.0)
    Uu = TrialFESpace(Vu, [u0])

    Uφᵛ = FESpace(Ωₕ, reffeφ, conformity=:H1)
    Uuv = FESpace(Ωₕ, reffeu, conformity=:H1)
    Γt1 = BoundaryTriangulation(model, tags="t1")
    Γt2 = BoundaryTriangulation(model, tags="t2")
    Γt3 = BoundaryTriangulation(model, tags="t3")
    Γt4 = BoundaryTriangulation(model, tags="t4")
    Γt5 = BoundaryTriangulation(model, tags="t5")
    Γt6 = BoundaryTriangulation(model, tags="t6")
    Γt7 = BoundaryTriangulation(model, tags="t7")
    Γt8 = BoundaryTriangulation(model, tags="t8")
    Γt9 = BoundaryTriangulation(model, tags="t9")
    Γt10 = BoundaryTriangulation(model, tags="t10")
    Γb1 = BoundaryTriangulation(model, tags="b1")
    Γb2 = BoundaryTriangulation(model, tags="b2")
    Γb3 = BoundaryTriangulation(model, tags="b3")
    Γb4 = BoundaryTriangulation(model, tags="b4")
    Γb5 = BoundaryTriangulation(model, tags="b5")
    Γb6 = BoundaryTriangulation(model, tags="b6")
    Γb7 = BoundaryTriangulation(model, tags="b7")
    Γb8 = BoundaryTriangulation(model, tags="b8")
    Γb9 = BoundaryTriangulation(model, tags="b9")
    Γb10 = BoundaryTriangulation(model, tags="b10")
    Uφˢt1 = FESpace(Γt1, reffeφ)
    Uuˢt1 = FESpace(Γt1, reffeu)
    Uφˢt2 = FESpace(Γt2, reffeφ)
    Uφˢt3 = FESpace(Γt3, reffeφ)
    Uφˢt4 = FESpace(Γt4, reffeφ)
    Uφˢt5 = FESpace(Γt5, reffeφ)
    Uφˢt6 = FESpace(Γt6, reffeφ)
    Uφˢt7 = FESpace(Γt7, reffeφ)
    Uφˢt8 = FESpace(Γt8, reffeφ)
    Uφˢt9 = FESpace(Γt9, reffeφ)
    Uφˢt10 = FESpace(Γt10, reffeφ)
    Uφˢb1 = FESpace(Γb1, reffeφ)
    Uφˢb2 = FESpace(Γb2, reffeφ)
    Uφˢb3 = FESpace(Γb3, reffeφ)
    Uφˢb4 = FESpace(Γb4, reffeφ)
    Uφˢb5 = FESpace(Γb5, reffeφ)
    Uφˢb6 = FESpace(Γb6, reffeφ)
    Uφˢb7 = FESpace(Γb7, reffeφ)
    Uφˢb8 = FESpace(Γb8, reffeφ)
    Uφˢb9 = FESpace(Γb9, reffeφ)
    Uφˢb10 = FESpace(Γb10, reffeφ)

    # Update Problem Parameters
    ndofm::Int = num_free_dofs(Vu)
    ndofe::Int = num_free_dofs(Vφ)
    Qₕ = CellQuadrature(Ωₕ, 4 * 2)
    fem_params = (; Ωₕ, dΩ, ndofm, ndofe, Uφᵛ,Uuv, Uφˢt1,Uuˢt1, Uφˢt2,Uφˢt3,Uφˢt4,Uφˢt5,Uφˢt6,Uφˢt7,Uφˢt8,Uφˢt9,Uφˢt10, Uφˢb1, Uφˢb2,Uφˢb3,Uφˢb4,Uφˢb5,Uφˢb6,Uφˢb7,Uφˢb8,Uφˢb9,Uφˢb10, Qₕ)

    N = VectorValue(0.0, 0.0, 1.0)
    Nh = interpolate_everywhere(N, Uu)


    # Setup non-linear solver
    nls = NLSolver(
        show_trace=true,
        method=:newton,
        iterations=20)

    solver = FESolver(nls)
    pvd_results = paraview_collection(result_folder*"results", append=false)
    #---------------------------------------------
    # State equation
    #---------------------------------------------
    # # Weak form
    function Mat_electro(uh::FEFunction)
        return (φ, vφ) -> ∫(∇(vφ) ⋅ (∂Ψφ ∘ (∇(uh), ∇(φ)))) * dΩ
    end

    function res_state((u, φ), (v, vφ))
        return ∫((∇(v)' ⊙ (∂Ψu ∘ (∇(u)', ∇(φ)))) + (∇(vφ)' ⋅ (∂Ψφ ∘ (∇(u)', ∇(φ))))) * dΩ
    end

    function jac_state((u, φ), (du, dφ), (v, vφ))
        return ∫(∇(v)' ⊙ (inner42 ∘ ((∂Ψuu ∘ (∇(u)', ∇(φ))), ∇(du)')) +
                ∇(dφ) ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(u)', ∇(φ))), ∇(v)')) +
                ∇(vφ)' ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(u)', ∇(φ))), ∇(du)')) +
                ∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ))) ⋅ ∇(dφ))) * dΩ
    end

    function StateEquationIter(target_gen, x0, ϕ_app, loadinc, ndofm, cache)
        #----------------------------------------------
        #Define trial FESpaces from Dirichlet values
        #----------------------------------------------
        Uφ = TrialFESpace(Vφ, [ϕ_app[1],ϕ_app[2],ϕ_app[3],ϕ_app[4],ϕ_app[5],ϕ_app[6],ϕ_app[7],ϕ_app[8],ϕ_app[9],ϕ_app[10],0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ϕ_app[11],ϕ_app[12],ϕ_app[13],ϕ_app[14],ϕ_app[15],ϕ_app[16],ϕ_app[17],ϕ_app[18],ϕ_app[19],ϕ_app[20]])
        U = MultiFieldFESpace([Uu, Uφ])    
        #----------------------------------------------
        #Update Dirichlet values solving electro problem
        #----------------------------------------------    
        x0_old = copy(x0)
        uh = FEFunction(Uu, x0[1:ndofm])
        lφ(vφ) = 0.0
        opφ = AffineFEOperator(Mat_electro(uh), lφ, Uφ, Vφ)
        φh = solve(opφ)
        x0[ndofm+1:end] = get_free_dof_values(φh)
        ph = FEFunction(U, x0)
        #----------------------------------------------
        #Coupled FE problem
        #----------------------------------------------
        op = FEOperator(res_state, jac_state, U, V)
        # loadfact = round(φap / φmax, digits=2)
        println("+++ Loadinc is  $loadinc    +++\n")
        cacheold = cache
        ph, cache = solve!(ph, solver, op, cache)
        flag::Bool = (cache.result.f_converged || cache.result.x_converged)
        #----------------------------------------------
        #Check convergence
        #----------------------------------------------
        if (flag == true)
            #writevtk(Ωₕ, "results/ex10/results_$(loadinc)", cellfields=["uh" => ph[1], "phi" => ph[2]])
            if (target_gen == 1)
            #pvd_results[loadinc] = createvtk(Ωₕ,result_folder * "Target_0$loadinc.vtu", cellfields=["uh" => ph[1], "phi" => ph[2]],order=2)
            else
            #pvd_results[loadinc] = createvtk(Ωₕ,result_folder * "Opti_0$loadinc.vtu", cellfields=["uh" => ph[1], "phi" => ph[2]],order=2)
            end
            return get_free_dof_values(ph), cache, flag
        else
            return x0_old, cacheold, flag 
        end
    end
    function StateEquation(target_gen,ϕ_app::Vector; fem_params)
        nsteps =  20
        Λ_inc = 1.0 / nsteps
        x0 = zeros(Float64, num_free_dofs(V))
        cache = nothing
        Λ     = 0.0
        loadinc = 0
        maxbisect = 10
        nbisect = 0
        while Λ < 1.0 - 1e-6
            Λ += Λ_inc
            Λ = min(1.0, Λ)
            x0, cache, flag  = StateEquationIter(target_gen, x0,Λ*ϕ_app, loadinc, fem_params.ndofm, cache)
            if (flag == true)
                u_Fe_Function = FEFunction(fem_params.Uuv, x0[1:fem_params.ndofm]) # Convierte a una FE
                u_Projected = interpolate_everywhere(u_Fe_Function, fem_params.Uuˢt1) #Interpola en una superficie la FE
                u_Vector_on_Surface = get_free_dof_values(u_Projected) # Saca un vector
                f_mat(x) = x
                X_projected = interpolate_everywhere(f_mat, fem_params.Uuˢt1)
                X_Vector_on_Surface = get_free_dof_values(X_projected)
                cd("Complex_Potential $folder_name")
                filename = string.(round.(Λ*ϕ_app,digits=4))
                open("$filename.txt","w") do io
                    writedlm(io,u_Vector_on_Surface)
                end
                # open("mat_coords.txt","w") do io
                #     writedlm(io,X_Vector_on_Surface)
                # end
                cd(dirname(@__FILE__))
            end
            
            if (flag == false)
                Λ    -= Λ_inc
                Λ_inc = Λ_inc / 2
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



    # ----------------------------
    # We generate the target
    # ----------------------------
    printstyled("--------------------------------\n"; color=:yellow)
    printstyled("Computing Solution \n"; color = :yellow)
    printstyled("--------------------------------\n";color = :yellow)
    xstate = StateEquation(1,input_potential; fem_params)


end



input = readdlm("LHS_Complex.txt")


#-------------------------------------------
# Multithreading using Threads and Channels
#-------------------------------------------

# Create a channel to hold the calls for each column of the input vector

# taskref = Ref{Task}() # This is usefull to later call the channel's status
# Ch = Channel(200;taskref=taskref, spawn=true) do c
#     vec = take!(c)
#     CompactCall(vec[1],vec[2])
# end

#mkdir("Complex_Test")
#CompactCall([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"Complex_test")
 for column in range(1,size(input)[2])
     vector = input[:,column]
     print(vector)
     folder_name = string.(vector)
     mkdir("Complex_Potential $folder_name")
     CompactCall(vector, folder_name); # Running in 20 steps. If there are no cutbacks, we should end up with 4000 results

 end


# for column in range(1,size(input)[2])
#     vector = input[:,column]
#     folder_name = string.(vector)
#     local input_argument = [vector, folder_name]
#     mkdir("Potential $folder_name")
#     put!(Ch,input_argument);

# end

