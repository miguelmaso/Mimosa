using Pkg
Pkg.activate(".")

using Gridap
using GridapGmsh
using Gridap.TensorValues
using ForwardDiff
using Mimosa
using NLopt
using WriteVTK


# Initialisation result folder
mesh_file = "./models/mesh_platebeam_elec.msh"
result_folder = "./results/ex6/"
setupfolder(result_folder)

# Material parameters
const λ = 10.0
const μ = 1.0
const ε = 1.0

# Kinematics
F(∇u) = one(∇u) + ∇u
J(F) = det(F)
H(F) = J(F) * inv(F)'
E(∇φ) = -∇φ
HE(∇u, ∇φ) = H(F(∇u)) * E(∇φ)
HEHE(∇u, ∇φ) = HE(∇u, ∇φ) ⋅ HE(∇u, ∇φ)
Ψm(∇u) = μ / 2 * tr((F(∇u))' * F(∇u)) - μ * logreg(J(F(∇u))) + (λ / 2) * (J(F(∇u)) - 1)^2
Ψe(∇u, ∇φ) = (-ε / (2 * J(F(∇u)))) * HEHE(∇u, ∇φ)
Ψ(∇u, ∇φ) = Ψm(∇u) + Ψe(∇u, ∇φ)

∂Ψ_∂∇u(∇u, ∇φ) = ForwardDiff.gradient(∇u -> Ψ(∇u, get_array(∇φ)), get_array(∇u))
∂Ψ_∂∇φ(∇u, ∇φ) = ForwardDiff.gradient(∇φ -> Ψ(get_array(∇u), ∇φ), get_array(∇φ))
∂2Ψ_∂2∇φ(∇u, ∇φ) = ForwardDiff.hessian(∇φ -> Ψ(get_array(∇u), ∇φ), get_array(∇φ))
∂2Ψ_∂2∇u(∇u, ∇φ) = ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u, get_array(∇φ)), get_array(∇u))
∂2Ψ_∂2∇φ∇u(∇u, ∇φ) = ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇φ(∇u, get_array(∇φ)), get_array(∇u))

∂Ψu(∇u, ∇φ) = TensorValue(∂Ψ_∂∇u(∇u, ∇φ))
∂Ψφ(∇u, ∇φ) = VectorValue(∂Ψ_∂∇φ(∇u, ∇φ))
∂Ψuu(∇u, ∇φ) = TensorValue(∂2Ψ_∂2∇u(∇u, ∇φ))
∂Ψφφ(∇u, ∇φ) = TensorValue(∂2Ψ_∂2∇φ(∇u, ∇φ))
∂Ψφu(∇u, ∇φ) = TensorValue(∂2Ψ_∂2∇φ∇u(∇u, ∇φ))

# Grid model
model = GmshDiscreteModel(mesh_file)
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "dirm_u0", [3])
add_tag_from_tags!(labels, "dire_mid", [1])
add_tag_from_tags!(labels, "dire_top", [2])
model_file = joinpath(result_folder, "model")
writevtk(model, model_file)


#Define Finite Element Collections
order = 1
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)

#Setup integration
degree = 2 * order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)

#Define Finite Element Spaces
Vu = TestFESpace(Ωₕ, reffeu, labels=labels, dirichlet_tags=["dirm_u0"], conformity=:H1)
Vφ = TestFESpace(Ωₕ, reffeφ, labels=labels, dirichlet_tags=["dire_mid", "dire_top"], conformity=:H1)
V = MultiFieldFESpace([Vu, Vφ])
u0 = VectorValue(0.0, 0.0, 0.0)
Uu = TrialFESpace(Vu, [u0])

Uφᵛ = FESpace(Ωₕ, reffeφ, conformity=:H1)
Γtop = BoundaryTriangulation(model, tags="dire_top")
Uφˢ = FESpace(Γtop, reffeφ)

# Update Problem Parameters
ndofm::Int = num_free_dofs(Vu)
ndofe::Int = num_free_dofs(Vφ)
Qₕ = CellQuadrature(Ωₕ, 4 * 2)
fem_params = (; Ωₕ, dΩ, ndofm, ndofe, Uφᵛ, Uφˢ, Qₕ)

N = VectorValue(0.0, 1.0, 0.0)
Nh = interpolate_everywhere(N, Uu)
uᵗ(x) = VectorValue([0.0, -((0.3 * 40.0) * (x[3] / 40.0)^2.0), 0.0])
opt_params = (; N, uᵗ)


# Setup non-linear solver
nls = NLSolver(
    show_trace=false,
    method=:newton,
    iterations=20)

solver = FESolver(nls)
pvd_results = paraview_collection(result_folder*"results", append=false)

#---------------------------------------------
# State equation
#---------------------------------------------
# # Weak form
function Mat_electro(uh::FEFunction)
    return (φ, vφ) -> ∫(∇(vφ) ⋅ (∂Ψφ ∘ (∇(uh), ∇(φ)))) * dΩ
end

function res_state((u, φ), (v, vφ))
    return ∫((∇(v)' ⊙ (∂Ψu ∘ (∇(u)', ∇(φ)))) + (∇(vφ)' ⋅ (∂Ψφ ∘ (∇(u)', ∇(φ))))) * dΩ
end

function jac_state((u, φ), (du, dφ), (v, vφ))
    return ∫(∇(v)' ⊙ (inner42 ∘ ((∂Ψuu ∘ (∇(u)', ∇(φ))), ∇(du)')) +
             ∇(dφ) ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(u)', ∇(φ))), ∇(v)')) +
             ∇(vφ)' ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(u)', ∇(φ))), ∇(du)')) +
             ∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(u)', ∇(φ))) ⋅ ∇(dφ))) * dΩ
end

function StateEquationIter(x0, φap, loadinc, ndofm, cache)
    #----------------------------------------------
    #Define trial FESpaces from Dirichlet values
    #----------------------------------------------
    Uφ = TrialFESpace(Vφ, [0.0, φap])
    U = MultiFieldFESpace([Uu, Uφ])
    #----------------------------------------------
    #Update Dirichlet values solving electro problem
    #----------------------------------------------
    x0_old = copy(x0)
    uh = FEFunction(Uu, x0[1:ndofm])
    lφ(vφ) = 0.0
    opφ = AffineFEOperator(Mat_electro(uh), lφ, Uφ, Vφ)
    φh = solve(opφ)
    x0[ndofm+1:end] = get_free_dof_values(φh)
    ph = FEFunction(U, x0)
    #----------------------------------------------
    #Coupled FE problem
    #----------------------------------------------
    op = FEOperator(res_state, jac_state, U, V)
    # loadfact = round(φap / φmax, digits=2)
    # println("+++ Loadinc $loadinc:  φap $φap in loadfact $loadfact +++\n")
    cacheold = cache
    ph, cache = solve!(ph, solver, op, cache)
    flag::Bool = (cache.result.f_converged || cache.result.x_converged)
    #----------------------------------------------
    #Check convergence
    #----------------------------------------------
    if (flag == true)
        # writevtk(Ωₕ, "results/ex6/results_$(loadinc)", cellfields=["uh" => ph[1], "phi" => ph[2]])
        return get_free_dof_values(ph), cache, flag
    else
        return x0_old, cacheold, flag
    end
end

function StateEquation(φmax::Float64; fem_params)
    nsteps = 30
    φ_inc = φmax / nsteps
    x0 = zeros(Float64, num_free_dofs(V))
    cache = nothing
    φap = 0.0
    loadinc = 0
    maxbisect = 10
    nbisect = 0
    while (φap / φmax) < 1.0 - 1e-6
        φap += φ_inc
        φap = min(φap, φmax)
        x0, cache, flag = StateEquationIter(x0, φap, loadinc, fem_params.ndofm, cache)
        if (flag == false)
            φap -= φ_inc
            φ_inc = φ_inc / 2
            nbisect += 1
        end
        if nbisect > maxbisect
            println("Maximum number of bisections reached")
            break
        end
        loadinc += 1
    end
    return x0
end

#---------------------------------------------
# Adjoint equation
#---------------------------------------------
# function Vec_adjoint(uh::FEFunction)
#     return (v,vφ)->∫(((uh - uᵗ)⋅Nh)*(Nh⋅v) + vφ*0.0)*dΩ
# end

function Mat_adjoint(uh::FEFunction, φh::FEFunction)
    return ((p, pφ), (v, vφ)) -> ∫(∇(v)' ⊙ (inner42 ∘ ((∂Ψuu ∘ (∇(uh)', ∇(φh))), ∇(p)')) +
                                   ∇(pφ) ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(uh)', ∇(φh))), ∇(v)')) +
                                   ∇(vφ)' ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(uh)', ∇(φh))), ∇(p)')) +
                                   ∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(uh)', ∇(φh))) ⋅ ∇(pφ))) * dΩ
end

function AdjointEquation(xstate, φmax; fem_params)
    u = xstate[1:fem_params.ndofm]
    φ = xstate[fem_params.ndofm+1:end]
    Uφ = TrialFESpace(Vφ, [0.0, φmax])
    uh = FEFunction(Uu, u)
    φh = FEFunction(Uφ, φ)
    Vec_adjoint((v, vφ)) = ∫(((uh - uᵗ) ⋅ Nh) * (Nh ⋅ v) + vφ * 0.0) * dΩ
    op = AffineFEOperator(Mat_adjoint(uh, φh), Vec_adjoint, V, V)
    kh = solve(op)
    return get_free_dof_values(kh)
end


#---------------------------------------------
# Objective Function
#---------------------------------------------

function 𝒥(xstate, φap; fem_params)
    u = xstate[1:fem_params.ndofm]
    φ = xstate[fem_params.ndofm+1:end]
    uh = FEFunction(Uu, u)
    Uφ = TrialFESpace(Vφ, [0.0, φap])
    φh = FEFunction(Uφ, φ)
    iter = numfiles("results/ex6") + 1
    obj = ∑(∫(0.5 * ((uh - uᵗ) ⋅ N) * ((uh - uᵗ) ⋅ N))Qₕ)
    println("Iter: $iter, 𝒥 = $obj")
    pvd_results[iter] = createvtk(fem_params.Ωₕ,result_folder * "_$iter.vtu", cellfields=["uh" => uh, "φh" => φh],order=2)

    # writevtk(fem_params.Ωₕ, "results/ex6/results_$(iter)", cellfields=["uh" => uh, "φh" => φh])
    return obj
end


#---------------------------------------------
# Derivatives
#---------------------------------------------

function Vec_descent(uh, φh, puh, pφh)
    return (vφ) -> ∫(-∇(vφ) ⋅ (inner32 ∘ ((∂Ψφu ∘ (∇(uh)', ∇(φh))), ∇(puh)')) -
                     ∇(vφ)' ⋅ ((∂Ψφφ ∘ (∇(uh)', ∇(φh))) ⋅ ∇(pφh))) * dΩ
end

function D𝒥Dφmax(x::Vector,xstate, xadjoint; fem_params, opt_params)

    φap = x[1] * opt_params.φmax
    u = xstate[1:fem_params.ndofm]
    φ = xstate[fem_params.ndofm+1:end]
    pu = xadjoint[1:fem_params.ndofm]
    pφ = xadjoint[fem_params.ndofm+1:end]

    Uφ = TrialFESpace(Vφ, [0.0, φap])
    uh = FEFunction(Uu, u)
    puh = FEFunction(Vu, pu)
    φh = FEFunction(Uφ, φ)
    pφh = FEFunction(Vφ, pφ)

    D𝒥Dφmaxᵛ = assemble_vector(Vec_descent(uh, φh, puh, pφh), fem_params.Uφᵛ) #Volumen
    D𝒥Dφmaxᵛₕ = FEFunction(fem_params.Uφᵛ, D𝒥Dφmaxᵛ) # Convierte a una FE
    D𝒥Dφmaxˢₕ = interpolate_everywhere(D𝒥Dφmaxᵛₕ, fem_params.Uφˢ) #Interpola en una superficie la FE
    D𝒥Dφmaxˢ = get_free_dof_values(D𝒥Dφmaxˢₕ) # Saca un vector

    return [sum(D𝒥Dφmaxˢ)]
end



#---------------------------------------------
# Initialization of optimization variables
#---------------------------------------------
φmax = 0.2
xini = [0.01]
grad = [0.0]
opt_params = (; N, uᵗ, φmax)

function fopt(x::Vector, grad::Vector; fem_params, opt_params)
    φap = x[1] * opt_params.φmax
    xstate = StateEquation(φap; fem_params)
    xadjoint = AdjointEquation(xstate, φap; fem_params)
    if length(grad) > 0
        dobjdΦ = D𝒥Dφmax(x, xstate, xadjoint; fem_params, opt_params)
        grad[:] = opt_params.φmax * dobjdΦ
    end
    fo = 𝒥(xstate, φap; fem_params)
    return fo
end

function electro_optimize(x_init; TOL=1e-4, MAX_ITER=500, fem_params, opt_params)
    ##################### Optimize #################
    opt = Opt(:LD_MMA, 1)
    opt.lower_bounds = 0
    opt.upper_bounds = 1
    opt.ftol_rel = TOL
    opt.maxeval = MAX_ITER
    opt.min_objective = (x0, grad) -> fopt(x0, grad; fem_params, opt_params)
    (f_opt, x_opt, ret) = optimize(opt, x_init)
    @show numevals = opt.numevals # the number of function evaluations
    return f_opt, x_opt, ret
end


# @time fopt(xini, grad; fem_params, opt_params)
 a, b, ret=electro_optimize(xini; TOL = 1e-8, MAX_ITER=500, fem_params, opt_params)
 vtk_save(pvd_results)
