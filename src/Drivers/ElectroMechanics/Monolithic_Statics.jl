
function execute(problem::ElectroMechProblem{:monolithic,:statics}; kwargs...)

    # Problem setting
    pname = _get_kwarg(:problemName, kwargs)
    ptype = "ElectroMechanics"
    soltype = "monolithic"
    regtype = "statics"
    ctype = CouplingStrategy{Symbol(soltype)}()
    printinfo = @dict ptype soltype regtype pname
    print_heading(printinfo)

    is_vtk = _get_kwarg(:is_vtk, kwargs, false)
    simdir_ = datadir("sims", pname)
    setupfolder(simdir_)
 
    # Constitutive models
    consmodel = _get_kwarg(:consmodel, kwargs)
    @assert consmodel isa ElectroMech

    # Derivatives
    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = consmodel(DerivativeStrategy{:analytic}())
    DΨ = @ntuple Ψ ∂Ψu ∂Ψφ ∂Ψuu ∂Ψφu ∂Ψφφ

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
    res((u, φ), (v, vφ)) = residual_EM(ctype, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), dΩ) # Add Neumann BC 
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

        post_params = @dict Ω is_vtk simdir_
        solver_params = @dict fe_spaces dirichletbc Ω dΩ DΨ res jac solveropt nlsolver post_params

        ph,cache = IncrementalSolver(problem, ctype, ph, solver_params)

    end
end

function ΔSolver!(problem::ElectroMechProblem{:monolithic,:statics}, ctype::CouplingStrategy{:monolithic}, ph, Λ, Λ_inc, params, cache)

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
    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ_inc)
    # Update Dirichlet for electro problem
    lφ(vφ) = -1.0 * residual_EM(CouplingStrategy{:staggered_E}(), (uh, φh), vφ, DΨ.∂Ψφ, dΩ)
    aφ(dφ, vφ) = jacobian_EM(CouplingStrategy{:staggered_E}(), (uh, φh), dφ, vφ, DΨ.∂Ψφφ, dΩ)
    opφ = AffineFEOperator(aφ, lφ, fe_spaces.Uφ, fe_spaces.Vφ)
    dφh = solve(opφ)

    pφh = get_free_dof_values(φh)
    pdφh = get_free_dof_values(dφh)
    pφh .+= pdφh

    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ)

    op = FEOperator(res, jac, fe_spaces.U, fe_spaces.V)
    ph, cache = solve!(ph, nlsolver, op, cache)

    return ph, cache
end


function computeOutputs!(::ElectroMechProblem{:monolithic,:statics}, pvd, ph, Λ, Λ_, post_params)

    println("STEP: $Λ_, LAMBDA: $Λ")
    println("============================")

    Ω = _get_kwarg(:Ω, post_params)
    is_vtk = _get_kwarg(:is_vtk, post_params)
    filePath = _get_kwarg(:simdir_, post_params)

    uh = ph[1]
    φh = ph[2]

    if is_vtk
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(
            Ω,
            filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u" => uh, "φ" => φh]
        )
    end

    return pvd
end