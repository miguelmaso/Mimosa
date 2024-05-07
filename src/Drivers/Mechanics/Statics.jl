
function execute(problem::MechanicalProblem{:statics}; kwargs...)
    

    # Problem setting
    pname = _get_kwarg(:problemName, kwargs)
    ptype = "Mechanics"
    regtype = "statics"
    printinfo = @dict ptype regtype pname
    print_heading(printinfo)

    is_vtk = _get_kwarg(:is_vtk, kwargs, false)
    simdir_ = datadir("sims", pname)
    setupfolder(simdir_)

    # Constitutive models
    consmodel = _get_kwarg(:consmodel, kwargs)
    @assert consmodel isa Mechano

    # Derivatives
    Ψ, ∂Ψu, ∂Ψuu = consmodel(DerivativeStrategy{:analytic}())
    DΨ = @ntuple Ψ ∂Ψu ∂Ψuu

    # grid model
    mesh_file = _get_kwarg(:meshfile, kwargs)
    model = GmshDiscreteModel(datadir("models", mesh_file))
    if is_vtk 
        writevtk(model, simdir_ * "/DiscreteModel")
    end

    # Setup integration
    order = _get_kwarg(:order, kwargs, 1)
    degree = 2 * order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
         
    # Dirichlet boundary conditions
    dirichletbc = _get_kwarg(:dirichletbc, kwargs)

    # Neumann boundary conditions
    neumannbc = _get_kwarg(:neumannbc, kwargs, nothing)

    if !isnothing(neumannbc)
        dΓ=Vector{Gridap.CellData.GenericMeasure}(undef, length(neumannbc.tags))
        for i in 1:length(neumannbc.tags)
            Γ= BoundaryTriangulation(model, tags=neumannbc.tags[i])
            dΓ[i]= Measure(Γ, degree)
        end
    end

    #  FE spaces
    fe_spaces = get_FE_spaces(problem, model, order, dirichletbc)

    # WeakForms

    function res(Λ, dΓ, neumannbc)
        res1(u, v) = residual_M(u,v,∂Ψu,dΩ)
        res2(v) = residual_Neumann(v,neumannbc, dΓ;  Λ=Λ)
        (u, v) -> res1(u, v)+res2(v)
    end
       
    jac(u, du, v) = jacobian_M(u, du, v, ∂Ψuu, dΩ)


    @timeit pname begin
        println("Defining Nonlinear solver")
        # NewtonRaphson solver
        solveropt = _get_kwarg(:solveropt, kwargs)
        nlsolver = get_FE_solver(solveropt)

        # Initialization
        x0 = zeros(Float64, num_free_dofs(fe_spaces.V))
        ph = FEFunction(fe_spaces.U, x0)
        @show size(get_free_dof_values(ph))


        post_params = @dict Ω is_vtk simdir_

        solver_params = @dict fe_spaces dirichletbc neumannbc Ω dΩ dΓ DΨ res jac solveropt nlsolver post_params
 
        ph = IncrementalSolver(problem, ph, solver_params)

     end


end

function ΔSolver!(problem::MechanicalProblem, ph, Λ, Λ_inc, params, cache)

    fe_spaces = _get_kwarg(:fe_spaces, params)
    dirichletbc = _get_kwarg(:dirichletbc, params)
    neumannbc = _get_kwarg(:neumannbc, params)
    res = _get_kwarg(:res, params)
    jac = _get_kwarg(:jac, params)
    nlsolver = _get_kwarg(:nlsolver, params)
    DΨ = _get_kwarg(:DΨ, params)
    dΩ = _get_kwarg(:dΩ, params)
    dΓ = _get_kwarg(:dΓ, params)

 
    # # Test and trial spaces for Λ_inc
    # fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ_inc)
    
    # # Update Dirichlet for electro problem
    # lφ(vφ) = -1.0 * residual_EM(CouplingStrategy{:staggered_E}(), (uh, φh), vφ, DΨ.∂Ψφ, dΩ)
    # aφ(dφ, vφ) = jacobian_EM(CouplingStrategy{:staggered_E}(), (uh, φh), dφ, vφ, DΨ.∂Ψφφ, dΩ)
    # opφ = AffineFEOperator(aφ, lφ, fe_spaces.Uφ, fe_spaces.Vφ)
    # dφh = solve(opφ)

    # pφh = get_free_dof_values(φh)
    # pdφh = get_free_dof_values(dφh)
    # pφh .+= pdφh

    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ)

    op = FEOperator(res(Λ, dΓ, neumannbc), jac, fe_spaces.U, fe_spaces.V)
    ph, cache = solve!(ph, nlsolver, op, cache)

    return ph, cache
end


 