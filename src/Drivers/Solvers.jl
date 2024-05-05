
function get_FE_solver(solveropt::Dict{Symbol,Real})
    nls_ = NLSolver(show_trace=solveropt[:nr_show_trace],
        method=:newton,
        iterations=solveropt[:nr_iter],
        ftol=solveropt[:nr_ftol])
    FESolver(nls_)
end


function IncrementalSolver(problem::Problem, ctype::CouplingStrategy{:monolithic}, ph::FEFunction, params::Dict{Symbol,Any})

    nsteps = _get_kwarg(:nsteps, params[:solveropt])
    maxbisec = _get_kwarg(:nbisec, params[:solveropt])
    filePath = _get_kwarg(:simdir_, params[:post_params])
    is_vtk = _get_kwarg(:is_vtk, params[:post_params])
    post_params = _get_kwarg(:post_params, params)

    pvd = paraview_collection(filePath * "/Results", append=false)

    Λ = 0.0
    Λ_inc = 1.0 / nsteps

    cache = nothing
    nbisect = 0
    ph_view = get_free_dof_values(ph)
    Λ_ = 0
    while Λ < 1.0 - 1e-6
        Λ += Λ_inc
        Λ = min(1.0, Λ)
        ph_ = copy(get_free_dof_values(ph))
        ph, cache = ΔSolver!(problem, ctype, ph, Λ, Λ_inc, params, cache)
        flag = (cache.result.f_converged || cache.result.x_converged)

        #Check convergence
        if (flag == true)
            Λ_ += 1
            # Write to PVD
            pvd = computeOutputs!(problem, pvd, ph, Λ, Λ_, post_params)
        else
            ph_view[:] = ph_
            # go back to previous ph
            Λ -= Λ_inc
            Λ_inc = Λ_inc / 2
            nbisect += 1
        end

        @assert(nbisect <= maxbisec, "Maximum number of bisections reached")

    end
    if is_vtk
        vtk_save(pvd)
    end
    return ph, cache
end

 
function update_velocity!(::ElectroMechProblem{:monolithic,:dynamics}, υh, ph, phold, Δt)
    ph_ = get_free_dof_values(ph[1])
    phold_ = get_free_dof_values(phold[1])
    υh = get_free_dof_values(υh)
    υh .*= -1.0
    υh .-= (2.0 / Δt) .* phold_
    υh .+= (2.0 / Δt) .* ph_
end


function update_velocity!(::MechanicalProblem{:dynamics}, υh, ph, phold, Δt)
    ph_ = get_free_dof_values(ph)
    phold_ = get_free_dof_values(phold)
    υh = get_free_dof_values(υh)
    υh .*= -1.0
    υh .-= (2.0 / Δt) .* phold_
    υh .+= (2.0 / Δt) .* ph_
end


function Midpoint_Timeintegrator(problem::Problem, ph::FEFunction, phold::FEFunction, υh::FEFunction, params::Dict{Symbol,Any})

    nsteps       = _get_kwarg(:nsteps, params[:solveropt])
    filePath     = _get_kwarg(:simdir_, params[:post_params])
    is_vtk       = _get_kwarg(:is_vtk, params[:post_params])
    post_params  = _get_kwarg(:post_params, params)
    res          = _get_kwarg(:res, params)
    jac          = _get_kwarg(:jac, params)
    dΩ           = _get_kwarg(:dΩ, params)
    ρ            = _get_kwarg(:ρ, params)
    fe_spaces    = _get_kwarg(:fe_spaces, params)
    dirichletbc  = _get_kwarg(:dirichletbc, params)
    nlsolver     = _get_kwarg(:nlsolver, params)

    pvd = paraview_collection(filePath * "/Results", append=false)

    t = 0.0
    Δt = params[:solveropt][:Δt]
    nsteps = params[:solveropt][:nsteps]
    cache = nothing
    phold_view = get_free_dof_values(phold)
    itime = 0
    KE = zeros(Float64, nsteps)
    EE = zeros(Float64, nsteps)

    for itime in 1:nsteps
        t += Δt
        residual = res(phold, υh, ρ, Δt, dΩ)
        jacobian = jac(ρ, Δt, dΩ)
        fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, t)
        op = FEOperator(residual, jacobian, fe_spaces.U, fe_spaces.V)
        ph, cache = solve!(ph, nlsolver, op, cache)
        update_velocity!(problem, υh, ph, phold, Δt)

        phold_view[:] = get_free_dof_values(ph)
        pvd, EE_ = computeOutputs!(problem, pvd, ph, t, itime, post_params)

        # Kinetic Energy
        KE[itime] = ∑(∫(0.5 * ρ * (υh ⋅ υh))dΩ)
        # Elastic Energy
        EE[itime] = ∑(EE_)

    end
    if is_vtk
        vtk_save(pvd)
    end
    return ph, KE, EE, cache
end

