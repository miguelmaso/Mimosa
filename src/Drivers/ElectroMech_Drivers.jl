
function execute(problem::ElectroMechProblem{:EMPlate}; kwargs...)
    println("Executing ElectroMechProblem{:EMPlate}")

    # Problem setting
    couplingstrategy = _get_kwarg(:couplingstrategy, kwargs, "monolithic")
    ctype = CouplingStrategy{Symbol(couplingstrategy)}()
    mesh_file = _get_kwarg(:model, kwargs, "EMPlate.msh")
    pname = "EMPlate"
    ptype = "ElectroMechanics"
    simdir_ = datadir("sims", pname)
    setupfolder(simdir_)

    # mechanical properties
    μ = _get_kwarg(:μ, kwargs, 1e6)
    λ = _get_kwarg(:λ, kwargs, 1e7)

    # electrical properties
    ε0 = _get_kwarg(:ε0, kwargs, 1.0)
    εr = _get_kwarg(:εr, kwargs, 4.0)
    ε = _get_kwarg(:ε, kwargs, 4.0)

    order = _get_kwarg(:order, kwargs, 1)

    printinfo = @dict pname ptype couplingstrategy mesh_file μ λ ε0 εr ε order
    print_heading(printinfo)

    # Constitutive models
    # modmec = NeoHookean3D(λ, μ)
    modmec = MoneyRivlin3D(λ, μ, 0.0)
    modelec = IdealDielectric(ε)
    modelEM = ElectroMech(modmec, modelec)

    # Derivatives
    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelEM(DerivativeStrategy{:analytic}())
    DΨ = @ntuple Ψ ∂Ψu ∂Ψφ ∂Ψuu ∂Ψφu ∂Ψφφ

    # grid model
    model = GmshDiscreteModel(datadir("models", mesh_file))
    labels = get_face_labeling(model)
    writevtk(model, simdir_ * "/DiscreteModel")

    # Setup integration
    degree = 2 * order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # Dirichlet boundary conditions
    dirichletbc_ = _get_kwarg(:dirichletbc, kwargs)
    dirichletbc = get_bc_func(dirichletbc_[:tags], dirichletbc_[:values])

    # FE spaces
    fe_spaces = get_FE_spaces(problem, ctype, model, order, dirichletbc)

    # WeakForms
    res((u, φ), (v, vφ)) = residual_EM(ctype, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), dΩ)
    jac((u, φ), (du, dφ), (v, vφ)) = jacobian_EM(ctype, (u, φ), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)

    @timeit pname begin
        println("Defining Nonlinear solver")
        # NewtonRaphson solver
        solveropt = _get_kwarg(:solveropt, kwargs)
        nlsolver = get_FE_solver(solveropt)

        # Initialization
        xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
        xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
        x0 = vcat(xu, xφ)
        ph = FEFunction(fe_spaces.U, x0)
        @show size(get_free_dof_values(ph))

        solver_params = @dict fe_spaces dirichletbc Ω dΩ DΨ res jac solveropt nlsolver 

        ph = Solver(problem, ctype, ph, solver_params)

     end


end

function ΔSolver!(problem::ElectroMechProblem, ctype::CouplingStrategy{:monolithic}, ph, Λ, Λ_inc, params, cache)

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


 