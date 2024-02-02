using Gridap
using Mimosa

mesh_file = "./models/crank.json"

# mesh_file = joinpath(dirname(@__FILE__), "../models/crank.json")
model = DiscreteModelFromFile(mesh_file)

order = 1
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
V0 = TestFESpace(model, reffe;
  conformity=:H1,
  dirichlet_tags=["surface_1", "surface_2"],
  dirichlet_masks=[(true, false, false), (true, true, true)])

g1(x) = VectorValue(0.005, 0.0, 0.0)
g2(x) = VectorValue(0.0, 0.0, 0.0)


U = TrialFESpace(V0, [g1, g2])

const E = 70.0e9
const ν = 0.33
const λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
const μ = E / (2 * (1 + ν))
σ(ε) = λ * tr(ε) * one(ε) + 2 * μ * ε

degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)


a(u, v) = ∫(ε(v) ⊙ (σ ∘ ε(u))) * dΩ
l(v) = 0

op = AffineFEOperator(a, l, U, V0)
uh = solve(op)

setupfolder("results/ex0")
writevtk(Ω, "results/ex0/results", cellfields=["uh" => uh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])



