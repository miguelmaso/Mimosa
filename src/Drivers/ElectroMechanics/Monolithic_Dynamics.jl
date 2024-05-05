
function execute(problem::ElectroMechProblem{:monolithic,:dynamics}; kwargs...)

    # Problem setting
    pname = _get_kwarg(:problemName, kwargs)
    ptype = "ElectroMechanics"
    soltype = "monolithic"
    regtype = "dynamics"

    ctype = CouplingStrategy{Symbol(soltype)}()
    printinfo = @dict ptype soltype regtype pname
    print_heading(printinfo)

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
    is_vtk = _get_kwarg(:is_vtk, kwargs, false)

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

    # # WeakForms
    function res(phold, υ, ρ, Δt,  dΩ)
        res1(u, v) = mass_term(u, v, 2.0 * ρ / Δt^2, dΩ)
        res2(v) = mass_term(phold[1], v, 2.0 * ρ / Δt^2, dΩ)
        res3(v) = mass_term(υ, v, 2.0 * ρ / Δt, dΩ)
        res4((u, φ), (v, vφ)) = residual_EM(ctype, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), dΩ)
        res5((v, vφ)) = residual_EM(ctype, (phold[1], phold[2]), (v, vφ), (∂Ψu, ∂Ψφ), dΩ)
        return ((u, φ), (v, vφ)) -> res1(u, v) - res2(v) - res3(v) + 0.5 * res4((u, φ), (v, vφ)) + 0.5 * res5((v, vφ))
    end

    function jac(ρ, Δt, dΩ)
        jac1(du, v) = mass_term(du, v, 2 * ρ / Δt^2, dΩ)
        jac2((u, φ), (du, dφ), (v, vφ)) = jacobian_EM(ctype, (u, φ), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)
        return ((u, φ), (du, dφ), (v, vφ)) -> jac1(du, v) + 0.5 * jac2((u, φ), (du, dφ), (v, vφ))
    end


      @timeit pname begin
        println("Begining mid-point solver")
        # NewtonRaphson solver
        solveropt = _get_kwarg(:solveropt, kwargs)
        nlsolver = get_FE_solver(solveropt)
        vel = _get_kwarg(:velocity, kwargs)

        # Initialization

        x0 = zeros(Float64, num_free_dofs(fe_spaces.V))
        ph = FEFunction(fe_spaces.U, x0)
        x0old = zeros(Float64, num_free_dofs(fe_spaces.V))
        phold = FEFunction(fe_spaces.U, x0old)
        υh = interpolate_everywhere(vel, fe_spaces.Vu)

        ρ = consmodel.Model1.ρ
        post_params = @dict Ω dΩ Ψ is_vtk simdir_
        solver_params = @dict fe_spaces dirichletbc Ω dΩ DΨ res jac ρ solveropt nlsolver post_params

        ph, KE, EE, cache = Midpoint_Timeintegrator(problem, ph, phold, υh, solver_params)

     end
end

  

function computeOutputs!(::ElectroMechProblem{:monolithic,:dynamics}, pvd, ph, Λ, Λ_, post_params)

    println("STEP: $Λ_, LAMBDA: $Λ")
    println("============================")

    Ω = _get_kwarg(:Ω, post_params)
    dΩ = _get_kwarg(:dΩ, post_params)
    Ψ = _get_kwarg(:Ψ, post_params)
    is_vtk = _get_kwarg(:is_vtk, post_params)
    filePath = _get_kwarg(:simdir_, post_params)

    uh = ph[1]
    φh = ph[2]

    if is_vtk && (Λ_%10==0)
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(
            Ω,
            filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u" => uh, "φ" => φh]
        )
    end


    # Internal Energy
    EE = ∫(Ψ ∘ (∇(ph[1])',∇(ph[2])'))dΩ

    return pvd, EE
end