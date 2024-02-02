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
mesh_file = "./models/mesh_platebeam_mag.msh"
result_folder = "./results/ex5/"
setupfolder(result_folder)

# Material parameters
const λ = 10.0
const μ = 1.0
const μ0 = 1.0

Bₐ = VectorValue(0.0, 0.0, 30.0e-5)

# Kinematics
F(∇u) = one(∇u) + ∇u
J(F) = det(F)
H(F) = J(F) * inv(F)'
FBr(∇u, Br) = F(∇u) * Br
FBr_Ba(∇u, Br, Ba) = (FBr(∇u, Br)) ⋅ Ba
Ψmec(∇u) = μ / 2 * tr((F(∇u))' * F(∇u)) - μ * logreg(J(F(∇u))) + (λ / 2) * (J(F(∇u)) - 1)^2
Ψmag(∇u, Br, Ba) = -μ0 * (FBr_Ba(∇u, Br, Ba))
Ψ(∇u, Br, Ba) = Ψmec(∇u) + Ψmag(∇u, Br, Ba)

∂Ψ_∂∇u(∇u, Br, Ba) = ForwardDiff.gradient(∇u -> Ψ(∇u, get_array(Br), get_array(Ba)), get_array(∇u))
∂2Ψ_∂2∇u(∇u, Br, Ba) = ForwardDiff.jacobian(∇u -> ∂Ψ_∂∇u(∇u, get_array(Br), get_array(Ba)), get_array(∇u))
∂2Ψ_∂2∇uBr(∇u, Br, Ba) = ForwardDiff.jacobian(Br -> ∂Ψ_∂∇u(∇u, get_array(Br), get_array(Ba)), get_array(Br))

∂Ψu(∇u, Br, Ba) = TensorValue(∂Ψ_∂∇u(∇u, Br, Ba))
∂Ψuu(∇u, Br, Ba) = TensorValue(∂2Ψ_∂2∇u(∇u, Br, Ba))
∂ΨuBr(∇u, Br, Ba) = TensorValue(∂2Ψ_∂2∇uBr(∇u, Br, Ba))

# Grid model
model = GmshDiscreteModel(mesh_file)
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "dirm_u0", [1])
model_file = joinpath(result_folder, "model")
writevtk(model, model_file)

#Define Finite Element Collections
order = 1
FEr = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
FE_L2 = ReferenceFE(lagrangian, Float64, 0)
FE_H1 = ReferenceFE(lagrangian, Float64, 1)
FE_H1_B = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)

#Setup integration
degree = 2 * order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)
nel = num_cells(Ωₕ)

#Define Finite Element Spaces
UΦ2 = FESpace(Ωₕ, FE_L2, vector_type=Vector{Float64}, conformity=:L2)
UΦ1 = FESpace(Ωₕ, FE_H1, vector_type=Vector{Float64}, conformity=:H1)
UB1 = FESpace(Ωₕ, FE_H1_B, vector_type=Vector{Float64}, conformity=:H1)
V = TestFESpace(model, FEr, labels=labels, dirichlet_tags=["dirm_u0"], conformity=:H1)
u0 = VectorValue(0.0, 0.0, 0.0)
U = TrialFESpace(V, [u0])

npt = num_free_dofs(UΦ1)
Qₕ = CellQuadrature(Ωₕ, 4 * 2)
fem_params = (; nel, npt, UΦ2, UΦ1, UB1, Ωₕ, dΩ, Qₕ)

Bah = interpolate_everywhere(Bₐ, V)
phys_params = (; Bₐ)

r = 1.3 * minimum(get_cell_measure(Ωₕ))
N = VectorValue(0.0, 0.0, 1.0)
Nh = interpolate_everywhere(N, U)
uᵗ(x) = ((0.1 * 40.0) * (x[1] / 40.0)^2.0)
opt_params = (; r, N, uᵗ)


a_f(r, u, v) = r^2 * (∇(v) ⋅ ∇(u))
function Filter(p0, r, fem_params)
    ph = FEFunction(fem_params.UΦ2, p0)
    op = AffineFEOperator(fem_params.UΦ1, fem_params.UΦ1) do u, v
        ∫(a_f(r, u, v))fem_params.dΩ + ∫(v * u)fem_params.dΩ, ∫(v * ph)fem_params.dΩ
    end
    pfh = solve(op)
    return get_free_dof_values(pfh)
end


function mapΦ_Br(Φ::Vector{Float64}; fem_params, opt_params)
    θ = Φ[1:fem_params.nel]
    φ = Φ[fem_params.nel+1:2*fem_params.nel]
    θf = Filter(θ, opt_params.r, fem_params)
    φf = Filter(φ, opt_params.r, fem_params)
    f(x, y) = [cos(x) * sin(y), sin(x) * sin(y), cos(y)] #mapΦ_Br
    f2(x, y) = [-sin(x) * sin(y), cos(x) * sin(y), 0.0] #map∂Br∂θ
    f3(x, y) = [cos(x) * cos(y), sin(x) * cos(y), -sin(y)] #map∂Br∂φ
    return reduce(vcat, f.(θf, φf))::Vector{Float64},
    reduce(vcat, f2.(θf, φf))::Vector{Float64},
    reduce(vcat, f3.(θf, φf))::Vector{Float64}
end

 

# Setup non-linear solver in State and Adjoint equations
nls = NLSolver(LUSolver(),
    show_trace=false,
    method=:newton)
solver = FESolver(nls)
pvd_results = paraview_collection(result_folder*"results", append=false)

#---------------------------------------------
# State equation
#---------------------------------------------
# # Weak form
function res_state(Bah::FEFunction, Brh::FEFunction)
    return (u, v) -> ∫((∇(v)' ⊙ (∂Ψu ∘ (∇(u)', Brh, Bah)))) * dΩ
end
function jac_state(Bah::FEFunction, Brh::FEFunction)
    return (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(u)', Brh, Bah))⊙ (∇(du)'))) * dΩ
end

 
function StateEquationIter(u, Bah, Brh, step, nsteps, cache)
    # Update FEFunction uh from vector u
    uh = FEFunction(U, u)
    #Update Dirichlet values FE problem
    op = FEOperator(res_state(Bah, Brh), jac_state(Bah, Brh), U, V)
    uh, cache = solve!(uh, solver, op, cache)
    return get_free_dof_values(uh), cache
end

function StateEquation(Φ; fem_params, phys_params, opt_params)
    Br, _, _ = mapΦ_Br(Φ; fem_params, opt_params)
    Brh = FEFunction(fem_params.UB1, Br)
    Bapp_inc = 1.0 / 40.0
    nsteps = ceil(Int, 1 / Bapp_inc)
    u = zeros(Float64, num_free_dofs(V))
    cache = nothing
    for step in 1:nsteps
        Λ = step / nsteps
        Bapp = phys_params.Bₐ * Λ
        Bah = interpolate_everywhere(Bapp, V)
        u, cache = StateEquationIter(u, Bah, Brh, step, nsteps, cache)
    end
    return u
end

#---------------------------------------------
# Adjoint equation
#---------------------------------------------

function Mat_adjoint(uh::FEFunction, Bah::FEFunction, Brh::FEFunction)
    return (p, v) -> ∫(∇(v)' ⊙ ((∂Ψuu ∘ (∇(uh)', Brh, Bah))⊙ (∇(p)'))) * dΩ
end
function Vec_adjoint(uh::FEFunction)
    return (v) -> ∫((uh ⋅ Nh - uᵗ) * (Nh ⋅ v)) * dΩ
end

function AdjointEquation(xstate, Φ; fem_params, opt_params)

    Br, _, _ = mapΦ_Br(Φ; fem_params, opt_params)
    Brh = FEFunction(fem_params.UB1, Br)
    uh = FEFunction(U, xstate)
    op = AffineFEOperator(Mat_adjoint(uh, Bah, Brh), Vec_adjoint(uh), V, V)
    ph = solve(op)
    return get_free_dof_values(ph)
end

#---------------------------------------------
# Objective Funciton equation
#---------------------------------------------
function 𝒥(u, fem_params)
    uh = FEFunction(U, u)
    iter = numfiles("results/ex5") + 1
    obj=∑(∫(0.5*(uh⋅Nh-uᵗ)*(uh⋅Nh- uᵗ))fem_params.Qₕ)
    println("Iter: $iter, 𝒥 = $obj")
    # writevtk(fem_params.Ωₕ, "results/ex5/results_$(iter)", cellfields=["uh" => uh])
    pvd_results[iter] = createvtk(fem_params.Ωₕ,result_folder * "_$iter.vtu", cellfields=["uh" => uh],order=2)
    return obj
end
 

#---------------------------------------------
# Derivatives
#---------------------------------------------
function Mat_descent(w, v)
    return ∫(a_f(r, w, v) + v * w) * dΩ
end

function Vec_descent(ph::FEFunction, uh::FEFunction, Brh::FEFunction, Bah::FEFunction, ∂Br::FEFunction)
    return (v) -> ∫(-(∇(ph)' ⊙  ((∂ΨuBr ∘ (∇(uh)', Brh, Bah))  ⊙  ∂Br)) * v) * dΩ
end

 
function D𝒥DΦ(Φ, u, p; fem_params, opt_params)

    Br, ∂Br_∂θ, ∂Br_∂φ = mapΦ_Br(Φ; fem_params, opt_params)

    Brh = FEFunction(fem_params.UB1, Br)
    ∂Br_∂θh = FEFunction(fem_params.UB1, ∂Br_∂θ)
    ∂Br_∂φh = FEFunction(fem_params.UB1, ∂Br_∂φ)

    uh = FEFunction(U, u)
    ph = FEFunction(V, p)

    opθ = AffineFEOperator(Mat_descent, Vec_descent(ph, uh, Brh, Bah, ∂Br_∂θh), fem_params.UΦ1, fem_params.UΦ1)
    ∂L_∂θfh = solve(opθ)
    opφ = AffineFEOperator(Mat_descent, Vec_descent(ph, uh, Brh, Bah, ∂Br_∂φh), fem_params.UΦ1, fem_params.UΦ1)
    ∂L_∂φfh = solve(opφ)
    DL_Dθ(v) = ∫(∂L_∂θfh * v) * dΩ
    DL_Dφ(v) = ∫(∂L_∂φfh * v) * dΩ
    ∂L_∂θ = assemble_vector(DL_Dθ, fem_params.UΦ2)
    ∂L_∂φ = assemble_vector(DL_Dφ, fem_params.UΦ2)
    return [∂L_∂θ; ∂L_∂φ]
end

#---------------------------------------------
# Initialization of optimization variables
#---------------------------------------------

function fopt(x::Vector, grad::Vector; fem_params, phys_params, opt_params)
    Φ = map(p -> (2.0 * pi) * (2.0 * p - 1.0), x)
    u = StateEquation(Φ; fem_params, phys_params, opt_params)
    p = AdjointEquation(u, Φ; fem_params, opt_params)
    if length(grad) > 0
        dobjdΦ = D𝒥DΦ(Φ, u, p; fem_params, opt_params)
        grad[:] = 4.0 * pi * dobjdΦ
    end
     fo = 𝒥(u, fem_params)
    return fo
end
 
θini = 0.5
φini = 0.5

xθ = fill((θini / (2.0 * pi) + 1.0) / 2.0, nel)
xφ = fill((φini / (2.0 * pi) + 1.0) / 2.0, nel)
x0 = [xθ; xφ]
grad = zeros(2 * nel)

function magnet_optimize(x_init; TOL=1e-4, MAX_ITER=500, fem_params, opt_params)
    ##################### Optimize #################
    opt = Opt(:LD_MMA, length(x_init))
    opt.lower_bounds = 0
    opt.upper_bounds = 1
    opt.ftol_rel = TOL
    opt.maxeval = MAX_ITER
    opt.min_objective = (x0, grad) -> fopt(x0, grad; fem_params, phys_params, opt_params)
    (f_opt, x_opt, ret) = optimize(opt, x_init)
    @show numevals = opt.numevals # the number of function evaluations
    return f_opt, x_opt, ret
end


  a, b, ret=magnet_optimize(x0; TOL = 1e-8, MAX_ITER=500, fem_params, opt_params)
# @show ret
vtk_save(pvd_results)
