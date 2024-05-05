
function execute(problem::MechanicalProblem{:dynamics}; kwargs...)

    # Problem setting
    pname = _get_kwarg(:problemName, kwargs)
    ptype = "Mechanics"
    regtype = "dynamics"

    print_heading(@dict pname ptype regtype)

    # Postprocessing
    is_vtk = _get_kwarg(:is_vtk, kwargs, false)
    simdir_ = datadir("sims", pname)
    setupfolder(simdir_)

    # Constitutive model
    consmodel = _get_kwarg(:consmodel, kwargs)
    @assert consmodel isa Mechano

    Ψ, ∂Ψu, ∂Ψuu = consmodel(DerivativeStrategy{:analytic}())
    DΨ = @ntuple Ψ ∂Ψu ∂Ψuu

    # grid model
    mesh_file = _get_kwarg(:mesh_file, kwargs)
    model = GmshDiscreteModel(datadir("models", mesh_file))
    if is_vtk
        writevtk(model, simdir_ * "/DiscreteModel")
    end

    # Setup integration
    order = _get_kwarg(:order, kwargs, 1)
    degree = 2 * order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # Get Dirichlet boundary conditions incremental functions
    dirichletbc = _get_kwarg(:dirichletbc, kwargs)

    # # Get Neumann boundary conditions incremental functions
    # neumannbc_ = _get_kwarg(:neumannbc, kwargs)
    # neumannbc = get_bc_func(neumannbc_[:tags], neumannbc_[:values])

    # # FE spaces
    fe_spaces = get_FE_spaces(problem, model, order, dirichletbc)

    # # WeakForms
    function res(uold, υ, ρ, Δt, dΩ)
        res1(u, v) = mass_term(u, v, 2.0 * ρ / Δt^2, dΩ)
        res2(v) = mass_term(uold, v, 2.0 * ρ / Δt^2, dΩ)
        res3(v) = mass_term(υ, v, 2.0 * ρ / Δt, dΩ)
        res4(u, v) = residual_M(u, v, ∂Ψu, dΩ)
        res5(v) = residual_M(uold, v, ∂Ψu, dΩ)
        return (u, v) -> res1(u, v) - res2(v) - res3(v) + 0.5 * res4(u, v) + 0.5 * res5(v)
    end

    function jac(ρ, Δt, dΩ)
        jac1(du, v) = mass_term(du, v, 2 * ρ / Δt^2, dΩ)
        jac2(u, du, v) = jacobian_M(u, du, v, ∂Ψuu, dΩ)
        return (u, du, v) -> jac1(du, v) + 0.5 * jac2(u, du, v)
    end

    @timeit pname begin
        println("Begining mid-point solver")
        # NewtonRaphson solver
        solveropt = _get_kwarg(:solveropt, kwargs)
        nlsolver = get_FE_solver(solveropt)
        vel = _get_kwarg(:velocity, kwargs)

        # Initialization
        x0 = zeros(Float64, num_free_dofs(fe_spaces.V))
        xold0 = zeros(Float64, num_free_dofs(fe_spaces.V))

        ph = FEFunction(fe_spaces.U, x0)
        phold = FEFunction(fe_spaces.U, xold0)
        υh = interpolate_everywhere(vel, fe_spaces.V)

        post_params = @dict Ω dΩ Ψ is_vtk simdir_
        ρ = consmodel.ρ
        solver_params = @dict fe_spaces dirichletbc Ω dΩ DΨ res jac ρ solveropt nlsolver post_params

        ph, KE, EE, cache = Midpoint_Timeintegrator(problem, ph, phold, υh, solver_params)

    end


end


function computeOutputs!(::MechanicalProblem{:dynamics}, pvd, ph, t, itime, post_params)

    println("STEP: $itime, Time: $t")
    println("============================")

    Ω = _get_kwarg(:Ω, post_params)
    dΩ = _get_kwarg(:dΩ, post_params)
    Ψ = _get_kwarg(:Ψ, post_params)

    is_vtk = _get_kwarg(:is_vtk, post_params)
    filePath = _get_kwarg(:simdir_, post_params)

    if is_vtk
        tstring = replace(string(round(itime, digits=2)), "." => "_")
        pvd[itime] = createvtk(
            Ω,
            filePath * "/_itime_" * tstring * "_TIME_$t" * ".vtu",
            cellfields=["u" => ph]
        )
    end

    # Internal Energy
    EE = ∫(Ψ ∘ (∇(ph)'))dΩ
    
    return pvd, EE
end