
function execute(problem::ThermoMechProblem{:monolithic,:dynamics}; kwargs...)

    # PhysicalProblem setting
    pname = _get_kwarg(:problemName, kwargs)
    ptype = "ThermoMechanics"
    soltype = "monolithic"
    regtype = "dynamics"

    printinfo = @dict ptype soltype regtype pname
    print_heading(printinfo)

    simdir_ = datadir("sims", pname)
    setupfolder(simdir_)
    
    is_vtk = _get_kwarg(:is_vtk, kwargs, false)

    # Constitutive models
    consmodel = _get_kwarg(:consmodel, kwargs)
    κ = consmodel.Thermo.κ
    θr = consmodel.Thermo.θr

    # Derivatives
    Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ ,η= consmodel(DerivativeStrategy{:analytic}())
    DΨ = @ntuple Ψ ∂Ψu ∂Ψθ ∂Ψuu ∂Ψθθ ∂Ψuθ 

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
    @assert dirichletbc isa MultiFieldBoundaryCondition
    @assert length(dirichletbc.BoundaryCondition)==2

    # Neumann boundary conditions
    neumannbc = _get_kwarg(:neumannbc, kwargs, MultiFieldBoundaryCondition([NothingBC(), NothingBC(),NothingBC()]) )
    @assert neumannbc isa MultiFieldBoundaryCondition || neumannbc isa NothingBC
    dΓ=get_Neumann_dΓ(model,neumannbc,degree)

    # FE spaces
    fe_spaces = get_FE_spaces(problem, model, order, dirichletbc)

    # # WeakForms
    solveropt = _get_kwarg(:solveropt, kwargs)
    αray = _get_kwarg(:αray, solveropt)


    function res(phold, υ, ρ, Δt,  dΩ)
        res1(u, v) = mass_term(u, v, 2.0 * ρ / Δt^2, dΩ)
        res2(v) = mass_term(phold[1], v, 2.0 * ρ / Δt^2, dΩ)
        res3(v) = mass_term(υ, v, 2.0 * ρ / Δt, dΩ)
        res4((u, θ), (v, vθ)) = residual(ThermoMechano, (u, θ), (v, vθ), ∂Ψu, κ, dΩ)
        res5((v, vθ)) = residual(ThermoMechano, (phold[1], phold[2]), (v, vθ), ∂Ψu, κ, dΩ)
        res6(u, v) = mass_term(u, v, αray * ρ / Δt, dΩ)
        res7(v) = mass_term(phold[1], v, αray * ρ / Δt, dΩ)
        res8((u, θ), vθ)= ∫((((η∘(∇(u)',  θ))-(η∘(∇(phold[1])', phold[2])))/Δt)* ((0.5*(θ+phold[2])+θr)⋅vθ))dΩ
        
        return ((u, θ), (v, vθ)) -> res1(u, v) - res2(v) - res3(v) + 0.5 * res4((u, θ), (v, vθ))+ 0.5 * res5((v, vθ))+res6(u, v)-res7(v)+res8((u, θ), vθ)
    end

    function jac(phold::FEFunction, ρ, Δt, dΩ)
        jac1(du, v) = mass_term(du, v, 2 * ρ / Δt^2, dΩ)
        jac2((u, θ), (du, dθ), (v, vθ)) = jacobian(ThermoMechano, (u,θ), (du, dθ), (v, vθ), (∂Ψuu, ∂Ψuθ), κ, dΩ)
        jac3(du, v) = mass_term(du, v, αray * ρ / Δt, dΩ)

        jac4((u,θ),(du, dθ) ,vθ)= ∫((-1.0/Δt)*((0.5*(θ+phold[2])+θr)⋅vθ)*((∂Ψuθ ∘ (∇(u)', θ)) ⊙ (∇(du)')))dΩ+
            ∫((-1.0/Δt)*((0.5*(θ+phold[2])+θr)⋅vθ)*((∂Ψθθ ∘  (∇(u)', θ)) *  dθ))dΩ+
            ∫((+1.0/(2*Δt))*((η∘(∇(u)', θ))-(η∘(∇(phold[1])', phold[2])))*dθ*vθ)dΩ

        return ((u, θ), (du, dθ), (v, vθ))  -> jac1(du, v) + 0.5 * jac2((u,θ), (du, dθ), (v, vθ)) +jac3(du, v) +jac4((u, θ),(du, dθ) ,vθ)
    end

    #   @timeit pname begin
        println("Begining mid-point solver")
        # NewtonRaphson solver
        nlsolver = get_FE_solver(solveropt)
        vel = _get_kwarg(:velocity, kwargs)

        # Initialization

        x0 = zeros(Float64, num_free_dofs(fe_spaces.V))
        ph = FEFunction(fe_spaces.U, x0)
        x0old = zeros(Float64, num_free_dofs(fe_spaces.V))
        phold = FEFunction(fe_spaces.U, x0old)
        υh = interpolate_everywhere(vel, fe_spaces.Vu)

        ρ = consmodel.Mechano.ρ
        post_params = @dict Ω dΩ Ψ is_vtk simdir_
        solver_params = @dict fe_spaces dirichletbc neumannbc Ω dΩ dΓ DΨ res jac ρ solveropt nlsolver post_params

        ph, KE, EE, cache = Midpoint_Timeintegrator(problem, ph, phold, υh, solver_params)

    #  end
end

  

function postprocess!(::ThermoMechProblem{:monolithic,:dynamics}, pvd, ph, Λ, Λ_, post_params)

    println("STEP: $Λ_, LAMBDA: $Λ")
    println("============================")

    Ω = _get_kwarg(:Ω, post_params)
    dΩ = _get_kwarg(:dΩ, post_params)
    Ψ = _get_kwarg(:Ψ, post_params)
    is_vtk = _get_kwarg(:is_vtk, post_params)
    filePath = _get_kwarg(:simdir_, post_params)

    uh = ph[1]
    θh = ph[2]

    if is_vtk && (Λ_%10==0)
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(
            Ω,
            filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["u" => uh,  "θ" => θh]
        )
    end


    # Internal Energy
    EE = ∫(Ψ ∘ (∇(ph[1])', ph[2]))dΩ

    return pvd, EE
end