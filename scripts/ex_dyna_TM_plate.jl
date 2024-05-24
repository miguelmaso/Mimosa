using DrWatson
using Mimosa
using Gridap
using Plots

function get_parameters()

  problemName = "dynamicTM"
  ptype = "ThermoMechanics"
  soltype = "monolithic"
  regtype = "dynamics"
  meshfile = "square.msh"

  # thermal properties
  β = 2.233e-4
  e = 5209.0

  # coupling parameters
  f(θ::Float64)::Float64 = 1.0
  df(θ::Float64)::Float64 = 0.0

  # Constitutive models
  modmec = MoneyRivlin3D(λ=10000.0, μ1=2.0e5, μ2=2.0e5, ρ=1.0)
  modterm = ThermalModel(Cv=100.0, θr=293.15, α=β * e, κ=10.0)
  consmodel = ThermoMech(modterm, modmec, f, df)

  # Boundary conditions 
  evolu(t) = 1.0
  dir_u_tags = ["fixed"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolθ(t) = 1.0
  dir_θ_tags = ["fixed"]
  dir_θ_values = [0.0]
  dir_θ_timesteps = [evolθ]
  Dθ = DirichletBC(dir_θ_tags, dir_θ_values, dir_θ_timesteps)

  dirichletbc = MultiFieldBoundaryCondition([Du, Dθ])

  evolθ_(t) = t<=1.0 ? sin(π*t/2) : 0.0
  neu_θ_tags = ["topsuf"]
  neu_θ_values = [(x)->((x[1]-50.0)^2 + (x[3]-50.0)^2 <= 25 ? 7.5e3/(pi*2.5^2) : 0.0)]
  neu_θ_timesteps = [evolθ_]
  Nθ = NeumannBC(neu_θ_tags, neu_θ_values, neu_θ_timesteps)

  neumannbc = MultiFieldBoundaryCondition([NothingBC(), Nθ])

  # initial conditions
  velocity(x) = VectorValue(0.0, 0.0, 0.0)

  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-3

  # Midpoint solver
  Δt = 0.1
  nsteps = 100
  αray = 0.0
  solveropt = @dict nr_show_trace nr_iter nr_ftol Δt nsteps αray

  # Postprocessing
  is_vtk = true

  return @dict problemName ptype soltype regtype meshfile consmodel dirichletbc neumannbc order solveropt is_vtk velocity
end

ph, KE, EE =main(; get_parameters()...)

plot(KE, label="Kinetic Energy")
plot!(EE, label="Elastic Energy")
plot!(KE + EE, label="Total Energy")

#  using PProf

#  PProf.Allocs.pprof(from_c=false)