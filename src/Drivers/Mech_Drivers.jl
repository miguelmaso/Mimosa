
function execute(problem::MechanicalProblem{:M_Plate}; kwargs...)
    

    # Problem setting
    mesh_file = _get_kwarg(:model, kwargs)
    pname = "M_Plate"
    ptype = "Mechanics"
    ctype = CouplingStrategy{Symbol("monolithic")}()
    simdir_ = datadir("sims", pname)
    setupfolder(simdir_)

    # mechanical properties
    μ = _get_kwarg(:μ, kwargs, 1e6)
    λ = _get_kwarg(:λ, kwargs, 1e7)
 
    order = _get_kwarg(:order, kwargs, 1)

    printinfo = @dict pname ptype ctype mesh_file μ λ order
    print_heading(printinfo)

    # Constitutive models
    modmec = MoneyRivlin3D(λ, μ, 0.0)

    # Derivatives
    Ψ, ∂Ψu, ∂Ψuu = modmec(DerivativeStrategy{:analytic}())
    DΨ = @ntuple Ψ ∂Ψu ∂Ψuu

    # grid model
    model = GmshDiscreteModel(datadir("models", mesh_file))
    labels = get_face_labeling(model)
    writevtk(model, simdir_ * "/DiscreteModel")

    # Setup integration
    degree = 2 * order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

 
     # Get Dirichlet boundary conditions incremental functions
    dirichletbc_ = _get_kwarg(:dirichletbc, kwargs)
    dirichletbc = get_bc_func( dirichletbc_[:tags], dirichletbc_[:values])

    # Get Neumann boundary conditions incremental functions
    neumannbc_ = _get_kwarg(:neumannbc, kwargs)
    neumannbc  = get_bc_func(neumannbc_[:tags], dirichletbc_[:values])

    # FE spaces
    fe_spaces = get_FE_spaces(problem, model, order, dirichletbc)

    # WeakForms
    res(u, v) = residual_M(u,v,∂Ψu,dΩ)
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

        solver_params = @dict fe_spaces dirichletbc Ω dΩ DΨ res jac solveropt nlsolver 

        # ph = IncrementalSolver(problem, ctype, ph, solver_params)

     end


end

function ΔSolver!(problem::MechanicalProblem, ctype::CouplingStrategy{:monolithic}, ph, Λ, Λ_inc, params, cache)

    fe_spaces = _get_kwarg(:fe_spaces, params)
    dirichletbc = _get_kwarg(:dirichletbc, params)
    res = _get_kwarg(:res, params)
    jac = _get_kwarg(:jac, params)
    nlsolver = _get_kwarg(:nlsolver, params)
    DΨ = _get_kwarg(:DΨ, params)
    dΩ = _get_kwarg(:dΩ, params)

    # update x0 with dirichlet incrementos   
    uh = ph[1] # not hard copy
    φh = ph[2] # not hard copy
    # Test and trial spaces for Λ_inc
    fe_spaces = get_FE_spaces!(problem, ctype, fe_spaces, dirichletbc, Λ_inc)
    # Update Dirichlet for electro problem
    lφ(vφ) = -1.0 * residual_EM(CouplingStrategy{:staggered_E}(), (uh, φh), vφ, DΨ.∂Ψφ, dΩ)
    aφ(dφ, vφ) = jacobian_EM(CouplingStrategy{:staggered_E}(), (uh, φh), dφ, vφ, DΨ.∂Ψφφ, dΩ)
    opφ = AffineFEOperator(aφ, lφ, fe_spaces.Uφ, fe_spaces.Vφ)
    dφh = solve(opφ)

    pφh = get_free_dof_values(φh)
    pdφh = get_free_dof_values(dφh)
    pφh .+= pdφh

    fe_spaces = get_FE_spaces!(problem, ctype, fe_spaces, dirichletbc, Λ)

    op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    ph, cache = solve!(ph, nlsolver, op, cache)

    return ph, cache
end


 