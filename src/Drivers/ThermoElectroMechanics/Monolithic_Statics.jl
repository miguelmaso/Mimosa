
function execute(problem::ThermoElectroMechProblem{:monolithic,:statics}; kwargs...)

    # Problem setting
    pname = _get_kwarg(:problemName, kwargs)
    ptype = "ThermoElectroMechanics"
    soltype = "monolithic"
    regtype = "statics"
    ctype = CouplingStrategy{Symbol(soltype)}()
    printinfo = @dict ptype soltype regtype pname
    print_heading(printinfo)

    simdir_ = datadir("sims", pname)
    setupfolder(simdir_)
    
    is_vtk = _get_kwarg(:is_vtk, kwargs, false)

    # Constitutive models
    consmodel = _get_kwarg(:consmodel, kwargs)
    κ = consmodel.Model1.κ

    # Derivatives
    Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ = consmodel(DerivativeStrategy{:analytic}())
    DΨ = @ntuple ∂Ψu ∂Ψφ ∂Ψθ ∂Ψuu ∂Ψφφ ∂Ψθθ ∂Ψφu ∂Ψuθ ∂Ψφθ

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

    # FE spaces
    fe_spaces = get_FE_spaces(problem, model, order, dirichletbc)

    # WeakForms
    res((u, φ, θ), (v, vφ, vθ)) = residual_TEM(ctype, (u, φ, θ), (v, vφ, vθ), (∂Ψu, ∂Ψφ), κ, dΩ)
    jac((u, φ, θ), (du, dφ, dθ), (v, vφ, vθ)) = jacobian_TEM(ctype, (u, φ, θ), (du, dφ, dθ), (v, vφ, vθ), (∂Ψuu, ∂Ψφφ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ), κ, dΩ)

    @timeit pname begin
        println("Defining Nonlinear solver")
        # NewtonRaphson solver
        solveropt = _get_kwarg(:solveropt, kwargs)
        nlsolver = get_FE_solver(solveropt)

        # Initialization
        xu = zeros(Float64, num_free_dofs(fe_spaces.Vu))
        xφ = zeros(Float64, num_free_dofs(fe_spaces.Vφ))
        xθ = zeros(Float64, num_free_dofs(fe_spaces.Vθ))
        x0 = vcat(xu, xφ, xθ)
        ph = FEFunction(fe_spaces.U, x0)

        post_params = @dict Ω is_vtk simdir_

        solver_params = @dict fe_spaces dirichletbc Ω dΩ DΨ κ res jac solveropt nlsolver post_params

        ph,cache = IncrementalSolver(problem, ctype, ph, solver_params)
    end


end


function ΔSolver!(problem::ThermoElectroMechProblem, ctype::CouplingStrategy{:monolithic}, ph, Λ, Λ_inc, params, cache)

    fe_spaces = _get_kwarg(:fe_spaces, params)
    dirichletbc = _get_kwarg(:dirichletbc, params)
    res = _get_kwarg(:res, params)
    jac = _get_kwarg(:jac, params)
    nlsolver = _get_kwarg(:nlsolver, params)
    DΨ = _get_kwarg(:DΨ, params)
    dΩ = _get_kwarg(:dΩ, params)
    κ = _get_kwarg(:κ, params)

    # update x0 with dirichlet incrementos   
    uh = ph[1] # not hard copy
    φh = ph[2] # not hard copy
    θh = ph[3] # not hard copy
    # Test and trial spaces for Λ_inc
    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ_inc)

    # Update Dirichlet for electro problem
    lφ(vφ) = -1.0 * residual_TEM(CouplingStrategy{:staggered_E}(), (uh, φh, θh), vφ, DΨ.∂Ψφ, dΩ)
    aφ(dφ, vφ) = jacobian_TEM(CouplingStrategy{:staggered_E}(), (uh, φh, θh), dφ, vφ, DΨ.∂Ψφφ, dΩ)
    opφ = AffineFEOperator(aφ, lφ, fe_spaces.Uφ, fe_spaces.Vφ)
    dφh = solve(opφ)

    # Update Dirichlet for thermal problem
    lθ(vθ) = -1.0 * residual_TEM(CouplingStrategy{:staggered_T}(), (uh, φh, θh), vθ, κ, dΩ)
    aθ(dθ, vθ) = jacobian_TEM(CouplingStrategy{:staggered_T}(), dθ, vθ, κ, dΩ)
    opθ = AffineFEOperator(aθ, lθ, fe_spaces.Uθ, fe_spaces.Vθ)
    dθh = solve(opθ)

    pφh = get_free_dof_values(φh)
    pdφh = get_free_dof_values(dφh)
    pφh .+= pdφh

    pθh = get_free_dof_values(θh)
    pdθh = get_free_dof_values(dθh)
    pθh .+= pdθh

    # Newton
    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ)

    op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    ph, cache = solve!(ph, nlsolver, op, cache)

    return ph, cache
end


function computeOutputs!(::ThermoElectroMechProblem{:monolithic, :statics}, pvd, ph, Λ, Λ_, post_params)

    println("STEP: $Λ_, LAMBDA: $Λ")
    println("============================")

    Ω = _get_kwarg(:Ω, post_params)
    is_vtk = _get_kwarg(:is_vtk, post_params)
    filePath = _get_kwarg(:simdir_, post_params)

    uh = ph[1]
    φh = ph[2]
    θh = ph[3]

    if is_vtk
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(
            Ω,
            filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u" => uh, "φ" => φh, "θ" => θh]
        )
    end

    return pvd
end