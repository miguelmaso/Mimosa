using Mimosa.WeakForms: jacobian
using Mimosa.WeakForms: residual

#------------------------------------------------------------
#                   Dirichlet Boundary conditions
#------------------------------------------------------------

function expand_DirichletBC!(problem::MechanicalProblem, ph, Λ, Λ_inc, params)
    fe_spaces = _get_kwarg(:fe_spaces, params)
    dirichletbc = _get_kwarg(:dirichletbc, params)
    DΨ = _get_kwarg(:DΨ, params)
    dΩ = _get_kwarg(:dΩ, params)
    neumannbc = _get_kwarg(:neumannbc, params)
    dΓ = _get_kwarg(:dΓ, params)

    # Test and trial spaces for Λ_inc
    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ_inc)

    # Update Mechanical Dirichlet 
    res_phys(v) = residual(Mechano, ph, v, DΨ.∂Ψu, dΩ)
    res_neu(v) = residual_Neumann(neumannbc, v, dΓ, Λ - Λ_inc)
    l(v) = -1.0 * (res_phys(v) + res_neu(v))
    a(du, v) = jacobian(Mechano, ph, du, v, DΨ.∂Ψuu, dΩ)
    op = AffineFEOperator(a, l, fe_spaces.U, fe_spaces.V)
    dph = solve(op)
    ph_ = get_free_dof_values(ph)
    dph_ = get_free_dof_values(dph)
    ph_ .+= dph_

end


function expand_DirichletBC!(problem::ElectroMechProblem{:monolithic,:statics}, ph, Λ, Λ_inc, params)
    fe_spaces = _get_kwarg(:fe_spaces, params)
    dirichletbc = _get_kwarg(:dirichletbc, params)
    DΨ = _get_kwarg(:DΨ, params)
    dΩ = _get_kwarg(:dΩ, params)
    neumannbc = _get_kwarg(:neumannbc, params)
    dΓ = _get_kwarg(:dΓ, params)

    # update x0 with dirichlet incrementos   
    uh = ph[1] # not hard copy
    φh = ph[2] # not hard copy
    # Test and trial spaces for Λ_inc
    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ_inc)
    # Update Dirichlet for electro problem
    res_physφ(vφ) = residual(ElectroMechano, Electro, (uh, φh), vφ, DΨ.∂Ψφ, dΩ)
    res_neuφ(vφ) = residual_Neumann(neumannbc.BoundaryCondition[2], vφ, dΓ[2], Λ - Λ_inc)
    lφ(vφ) = -1.0 * (res_physφ(vφ) + res_neuφ(vφ))
    aφ(dφ, vφ) = jacobian(ElectroMechano, Electro, (uh, φh), dφ, vφ, DΨ.∂Ψφφ, dΩ)
    opφ = AffineFEOperator(aφ, lφ, fe_spaces.Uφ, fe_spaces.Vφ)
    dφh = solve(opφ)

    pφh = get_free_dof_values(φh)
    pdφh = get_free_dof_values(dφh)
    pφh .+= pdφh


end


function expand_DirichletBC!(problem::ThermoElectroMechProblem{:monolithic,:statics}, ph, Λ, Λ_inc, params)
    fe_spaces = _get_kwarg(:fe_spaces, params)
    dirichletbc = _get_kwarg(:dirichletbc, params)
    DΨ = _get_kwarg(:DΨ, params)
    dΩ = _get_kwarg(:dΩ, params)
    κ = _get_kwarg(:κ, params)
    neumannbc = _get_kwarg(:neumannbc, params)
    dΓ = _get_kwarg(:dΓ, params)

    # update x0 with dirichlet incrementos   
    uh = ph[1] # not hard copy
    φh = ph[2] # not hard copy
    θh = ph[3] # not hard copy
    # Test and trial spaces for Λ_inc
    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ_inc)

    # Update Dirichlet for electro problem
    res_physφ(vφ) = residual(ThermoElectroMechano, Electro, (uh, φh, θh), vφ, DΨ.∂Ψφ, dΩ)
    res_neuφ(vφ) = residual_Neumann(neumannbc.BoundaryCondition[2], vφ, dΓ[2], Λ - Λ_inc)
    lφ(vφ) = -1.0 * (res_physφ(vφ) + res_neuφ(vφ))
    aφ(dφ, vφ) = jacobian(ThermoElectroMechano,Electro, (uh, φh, θh), dφ, vφ, DΨ.∂Ψφφ, dΩ)
    opφ = AffineFEOperator(aφ, lφ, fe_spaces.Uφ, fe_spaces.Vφ)
    dφh = solve(opφ)

    # Update Dirichlet for thermal problem
    res_physθ(vθ) = residual(ThermoElectroMechano, Thermo, (uh, φh, θh), vθ, κ, dΩ)
    res_neuθ(vθ) = residual_Neumann(neumannbc.BoundaryCondition[3], vθ, dΓ[3], Λ - Λ_inc)
    lθ(vθ) = -1.0 * (res_physθ(vθ) + res_neuθ(vθ))
    aθ(dθ, vθ) = jacobian(ThermoElectroMechano, Thermo, dθ, vθ, κ, dΩ)
    opθ = AffineFEOperator(aθ, lθ, fe_spaces.Uθ, fe_spaces.Vθ)
    dθh = solve(opθ)

    pφh = get_free_dof_values(φh)
    pdφh = get_free_dof_values(dφh)
    pφh .+= pdφh

    pθh = get_free_dof_values(θh)
    pdθh = get_free_dof_values(dθh)
    pθh .+= pdθh

end


function expand_DirichletBC!(problem::ThermoMechProblem{:monolithic,:statics}, ph, Λ, Λ_inc, params)
    fe_spaces = _get_kwarg(:fe_spaces, params)
    dirichletbc = _get_kwarg(:dirichletbc, params)
    DΨ = _get_kwarg(:DΨ, params)
    dΩ = _get_kwarg(:dΩ, params)
    κ = _get_kwarg(:κ, params)
    neumannbc = _get_kwarg(:neumannbc, params)
    dΓ = _get_kwarg(:dΓ, params)

    # update x0 with dirichlet incrementos   
    uh = ph[1] # not hard copy
    θh = ph[2] # not hard copy
    # Test and trial spaces for Λ_inc
    fe_spaces = get_FE_spaces!(problem, fe_spaces, dirichletbc, Λ_inc)

    # Update Mechanical Dirichlet 
    res_phys(v) = residual(ThermoMechano, Mechano, (uh, θh), v, DΨ.∂Ψu, dΩ)
    res_neu(v) = residual_Neumann(neumannbc.BoundaryCondition[1], v, dΓ[1], Λ - Λ_inc)
    l(v) = -1.0 * (res_phys(v) + res_neu(v))
    a(du, v) = jacobian(ThermoMechano, Mechano, (uh, θh), du, v, DΨ.∂Ψuu, dΩ)
    op = AffineFEOperator(a, l, fe_spaces.Uu, fe_spaces.Vu)
    duh = solve(op)
    puh = get_free_dof_values(uh)
    pduh = get_free_dof_values(duh)
    puh .+= pduh


    # Update Dirichlet for thermal problem
    res_physθ(vθ) = residual(ThermoMechano, Thermo, (uh, θh), vθ, κ, dΩ)
    res_neuθ(vθ) = residual_Neumann(neumannbc.BoundaryCondition[2], vθ, dΓ[2], Λ - Λ_inc)
    lθ(vθ) = -1.0 * (res_physθ(vθ) + res_neuθ(vθ))
    aθ(dθ, vθ) = jacobian(ThermoMechano, Thermo, dθ, vθ, κ, dΩ)
    opθ = AffineFEOperator(aθ, lθ, fe_spaces.Uθ, fe_spaces.Vθ)
    dθh = solve(opθ)
    pθh = get_free_dof_values(θh)
    pdθh = get_free_dof_values(dθh)
    pθh .+= pdθh

end


#------------------------------------------------------------
#                   Neumann Boundary conditions
#------------------------------------------------------------

function add_Neumann(::PhysicalProblem, residual, bc::NothingBC, dΓ; kwargs...)
    residual
end


function add_Neumann(::MechanicalProblem{:statics}, residual, bc::NeumannBC, dΓ; Λ=1.0)
    res_neu(u, v) = residual_Neumann(bc, v, dΓ, Λ)
    (u, v) -> residual(u, v) + res_neu(u, v)
end
function add_Neumann(::MechanicalProblem{:dynamics}, residual, bc::NeumannBC, dΓ; t⁺=1.0, t⁻=0.0)
    res_neu(u, v) = residual_Neumann(bc, v, dΓ, t⁺, t⁻)
    (u, v) -> residual(u, v) + res_neu(u, v)
end



function add_Neumann(::ElectroMechProblem{:monolithic,:statics}, residual, bc::MultiFieldBoundaryCondition, dΓ; Λ=1.0)
    res_neu((u, φ), (v, vφ)) = residual_Neumann(bc.BoundaryCondition[1], v, dΓ[1], Λ) + residual_Neumann(bc.BoundaryCondition[2], vφ, dΓ[2], Λ)
    ((u, φ), (v, vφ)) -> residual((u, φ), (v, vφ)) + res_neu((u, φ), (v, vφ))
end

function add_Neumann(::ElectroMechProblem{:monolithic,:dynamics}, residual, bc::MultiFieldBoundaryCondition, dΓ; t⁺=1.0, t⁻=0.0)
    res_neu((u, φ), (v, vφ)) = residual_Neumann(bc.BoundaryCondition[1], v, dΓ[1], t⁺, t⁻) + residual_Neumann(bc.BoundaryCondition[2], vφ, dΓ[2], t⁺, t⁻)
    ((u, φ), (v, vφ)) -> residual((u, φ), (v, vφ)) + res_neu((u, φ), (v, vφ))
end

function add_Neumann(::ThermoMechProblem{:monolithic,:statics}, residual, bc::MultiFieldBoundaryCondition, dΓ; Λ=1.0)
    res_neu((u, θ), (v, vθ)) = residual_Neumann(bc.BoundaryCondition[1], v, dΓ[1], Λ) +
                               residual_Neumann(bc.BoundaryCondition[2], vθ, dΓ[2], Λ)

    ((u, θ), (v, vθ)) -> residual((u, θ), (v, vθ)) + res_neu((u, θ), (v, vθ))
end

function add_Neumann(::ThermoMechProblem{:monolithic,:dynamics}, residual, bc::MultiFieldBoundaryCondition, dΓ; t⁺=1.0, t⁻=0.0)
    res_neu((u, θ), (v, vθ)) = residual_Neumann(bc.BoundaryCondition[1], v, dΓ[1], t⁺, t⁻) +
                                      residual_Neumann(bc.BoundaryCondition[2], vθ, dΓ[2], t⁺, t⁻)

    ((u, θ), (v, vθ)) -> residual((u, θ), (v, vθ)) + res_neu((u, θ), (v, vθ))
end



function add_Neumann(::ThermoElectroMechProblem{:monolithic,:statics}, residual, bc::MultiFieldBoundaryCondition, dΓ; Λ=1.0)
    res_neu((u, φ, θ), (v, vφ, vθ)) = residual_Neumann(bc.BoundaryCondition[1], v, dΓ[1], Λ) +
                                      residual_Neumann(bc.BoundaryCondition[2], vφ, dΓ[2], Λ) +
                                      residual_Neumann(bc.BoundaryCondition[3], vθ, dΓ[3], Λ)

    ((u, φ, θ), (v, vφ, vθ)) -> residual((u, φ, θ), (v, vφ, vθ)) + res_neu((u, φ, θ), (v, vφ, vθ))
end

function add_Neumann(::ThermoElectroMechProblem{:monolithic,:dynamics}, residual, bc::MultiFieldBoundaryCondition, dΓ; t⁺=1.0, t⁻=0.0)
    res_neu((u, φ, θ), (v, vφ, vθ)) = residual_Neumann(bc.BoundaryCondition[1], v, dΓ[1], t⁺, t⁻) +
                                      residual_Neumann(bc.BoundaryCondition[2], vφ, dΓ[2], t⁺, t⁻) +
                                      residual_Neumann(bc.BoundaryCondition[3], vθ, dΓ[3], t⁺, t⁻)

    ((u, φ, θ), (v, vφ, vθ)) -> residual((u, φ, θ), (v, vφ, vθ)) + res_neu((u, φ, θ), (v, vφ, vθ))
end
