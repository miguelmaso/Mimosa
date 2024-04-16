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


# Initialisation result folder
#mesh_file = "../examples/ex6/parametrize_plate_elec.msh"
mesh_file = joinpath(dirname(@__FILE__), "parametrize_plate_elec.msh")

result_folder = "./results/Data_Results/"
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



    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "fix", [7])
    add_tag_from_tags!(labels, "b1", [1])
    add_tag_from_tags!(labels, "b2", [2])
    add_tag_from_tags!(labels, "m1", [3])
    add_tag_from_tags!(labels, "m2", [4])
    add_tag_from_tags!(labels, "t1", [5])
    add_tag_from_tags!(labels, "t2", [6])
    model_file = joinpath(result_folder, "model")
    writevtk(model, model_file)


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
    Vφ = TestFESpace(Ωₕ, reffeφ, labels=labels, dirichlet_tags=["t1", "t2","m1","m2","b1","b2"], conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    u0 = VectorValue(0.0, 0.0, 0.0)
    Uu = TrialFESpace(Vu, [u0])

    Uφᵛ = FESpace(Ωₕ, reffeφ, conformity=:H1)
    Γt1 = BoundaryTriangulation(model, tags="t1")
    Γt2 = BoundaryTriangulation(model, tags="t2")
    Γb1 = BoundaryTriangulation(model, tags="b1")
    Γb2 = BoundaryTriangulation(model, tags="b2")
    Uφˢt1 = FESpace(Γt1, reffeφ)
    Uφˢt2 = FESpace(Γt2, reffeφ)
    Uφˢb1 = FESpace(Γb1, reffeφ)
    Uφˢb2 = FESpace(Γb2, reffeφ)

    # Update Problem Parameters
    ndofm::Int = num_free_dofs(Vu)
    ndofe::Int = num_free_dofs(Vφ)
    Qₕ = CellQuadrature(Ωₕ, 4 * 2)
    fem_params = (; Ωₕ, dΩ, ndofm, ndofe, Uφᵛ, Uφˢt1, Uφˢt2, Uφˢb1, Uφˢb2, Qₕ)

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
        Uφ = TrialFESpace(Vφ, [ϕ_app[1],ϕ_app[2],0.0,0.0,ϕ_app[3],ϕ_app[4]])
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
            pvd_results[loadinc] = createvtk(Ωₕ,result_folder * "Target_0$loadinc.vtu", cellfields=["uh" => ph[1], "phi" => ph[2]],order=2)
            else
            pvd_results[loadinc] = createvtk(Ωₕ,result_folder * "Opti_0$loadinc.vtu", cellfields=["uh" => ph[1], "phi" => ph[2]],order=2)
            end
            return get_free_dof_values(ph), cache, flag
        else
            return x0_old, cacheold, flag 
        end
    end
    function StateEquation(target_gen,ϕ_app::Vector; fem_params)
        nsteps = 5
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
            u_Fe_Function = FEFunction(fem_params.Uφᵛ, x0[1:fem_params.ndofm]) # Convierte a una FE
            u_Projected = interpolate_everywhere(u_Fe_Function, fem_params.Uφˢt1) #Interpola en una superficie la FE
            u_Vector_on_Surface = get_free_dof_values(u_Projected) # Saca un vector
            cd("Potential $folder_name")
            filename = string.(round.(Λ*ϕ_app,digits=4))
            open("$filename.txt","w") do io
                writedlm(io,u_Vector_on_Surface)
            end
            cd(dirname(@__FILE__))
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


    # #---------------------------------------------
    # # Objective Function
    # #---------------------------------------------

    # function 𝒥(xstate, ϕ_app; fem_params)
    #     u = xstate[1:fem_params.ndofm]
    #     φ = xstate[fem_params.ndofm+1:end]
    #     uh = FEFunction(Uu, u)
    #     Uφ = TrialFESpace(Vφ, [ϕ_app[1],ϕ_app[2],0.0,0.0,ϕ_app[3],ϕ_app[4]])
    #     φh = FEFunction(Uφ, φ)
    #     iter = numfiles("results/ex10") + 1
    #     @show norm(get_free_dof_values(u_tt))
    #     obj = ∑(∫(0.5 * ((uh - u_tt) ⋅ N) * ((uh - u_tt) ⋅ N))Qₕ)
    #     println("Iter: $iter, 𝒥 = $obj")
    #     pvd_results[iter] = createvtk(fem_params.Ωₕ,result_folder * "_$iter.vtu", cellfields=["uh" => uh, "φh" => φh],order=2)

    #     # writevtk(fem_params.Ωₕ, "results/ex6/results_$(iter)", cellfields=["uh" => uh, "φh" => φh])
    #     return obj
    # end


    # #---------------------------------------------
    # # Derivatives
    # #---------------------------------------------

    # function Vec_descent(uh, φh, puh, pφh)
    #     return (vφ) -> ∫(-∇(vφ) ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(uh)', ∇(φh))), ∇(puh)')) -
    #                      ∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(uh)', ∇(φh))) ⋅ ∇(pφh))) * dΩ
    # end

    # function D𝒥Dφmax(x::Vector,xstate, xadjoint; fem_params, opt_params)

    #     ϕ_app = x * opt_params.ϕ_max
    #     u = xstate[1:fem_params.ndofm]
    #     φ = xstate[fem_params.ndofm+1:end]
    #     pu = xadjoint[1:fem_params.ndofm]
    #     pφ = xadjoint[fem_params.ndofm+1:end]

    #     Uφ = TrialFESpace(Vφ, [ϕ_app[1],ϕ_app[2],0.0,0.0,ϕ_app[3],ϕ_app[4]])
    #     uh = FEFunction(Uu, u)
    #     puh = FEFunction(Vu, pu)
    #     φh = FEFunction(Uφ, φ)
    #     pφh = FEFunction(Vφ, pφ)

    #     D𝒥Dφmaxᵛ = assemble_vector(Vec_descent(uh, φh, puh, pφh), fem_params.Uφᵛ) #Volumen
    #     D𝒥Dφmaxᵛₕ = FEFunction(fem_params.Uφᵛ, D𝒥Dφmaxᵛ) # Convierte a una FE
    #     D𝒥Dφmaxˢt1 = interpolate_everywhere(D𝒥Dφmaxᵛₕ, fem_params.Uφˢt1) #Interpola en una superficie la FE
    #     D𝒥Dφmaxˢt2 = interpolate_everywhere(D𝒥Dφmaxᵛₕ, fem_params.Uφˢt2) #Interpola en una superficie la FE
    #     D𝒥Dφmaxˢb1 = interpolate_everywhere(D𝒥Dφmaxᵛₕ, fem_params.Uφˢb1) #Interpola en una superficie la FE
    #     D𝒥Dφmaxˢb2 = interpolate_everywhere(D𝒥Dφmaxᵛₕ, fem_params.Uφˢb2) #Interpola en una superficie la FE
    #     D𝒥Dφmaxˢst1 = get_free_dof_values(D𝒥Dφmaxˢt1) # Saca un vector
    #     D𝒥Dφmaxˢst2 = get_free_dof_values(D𝒥Dφmaxˢt2) # Saca un vector
    #     D𝒥Dφmaxˢsb1 = get_free_dof_values(D𝒥Dφmaxˢb1) # Saca un vector
    #     D𝒥Dφmaxˢsb2 = get_free_dof_values(D𝒥Dφmaxˢb2) # Saca un vector

    #     return [sum(D𝒥Dφmaxˢst1),sum(D𝒥Dφmaxˢst2),sum(D𝒥Dφmaxˢsb1),sum(D𝒥Dφmaxˢsb2)]
    # end

    # ----------------------------
    # We generate the target
    # ----------------------------
    printstyled("--------------------------------\n"; color=:yellow)
    printstyled("Starting the target generation\n"; color = :yellow)
    printstyled("--------------------------------\n";color = :yellow)
    xstate = StateEquation(1,input_potential; fem_params)

    #---------------------------------------------------------------
    # We get the displacements and project them in a given surface
    #--------------------------------------------------------------
    u_Fe_Function = FEFunction(fem_params.Uφᵛ, xstate[1:fem_params.ndofm]) # Convierte a una FE
    u_Projected = interpolate_everywhere(u_Fe_Function, fem_params.Uφˢt1) #Interpola en una superficie la FE
    u_Vector_on_Surface = get_free_dof_values(u_Projected) # Saca un vector

   return u_Vector_on_Surface
end


input = [0.01, 0.0, 0.0, 0.01]

folder_name = string.(input)
mkdir("Potential $folder_name")
#cd("Potential $folder_name")

output = CompactCall(input, folder_name);


#TODO Naming para el archivo output. He pensado que puedes crear una carpeta que se llame
# como el vector de input y que, dentro de la carpeta, pongas los resultados  del StateEquation que te va sacando para cada loadstep
