using DrWatson
using Mimosa
using Gridap
using Plots

function get_parameters()

  problemName = "pruebecita"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "dynamics"
  meshfile = "square.msh"

  modmec = MoneyRivlin3D(λ=10000.0, μ1=2.0e5, μ2=2.0e5, ρ=1.0)
  modelec = IdealDielectric(ε=4*8.854e-12)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 
  evolu(t) = 1.0
  dir_u_tags = ["fixed"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolφ(t) = t<=1.0 ? sin(π*t/2) : 1.0
  dir_φ_tags = ["midsuf", "topsuf"]
  dir_φ_values = [0.0, 1.0e7]
  dir_φ_timesteps = [evolφ, evolφ]
  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])

  # initial conditions
  velocity(x) = VectorValue(0.0, 0.0, 0.0)

   # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-3

  # Midpoint solver
  Δt = 0.005
  nsteps = 100
  αray = 0.0
  solveropt = @dict nr_show_trace nr_iter nr_ftol Δt nsteps αray

  # Postprocessing
  is_vtk = true

  return @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk velocity
end

ph, KE, EE = main(; get_parameters()...)

plot(KE, label="Kinetic Energy")
plot!(EE, label="Elastic Energy")
plot!(KE + EE, label="Total Energy")

#  using PProf

#  PProf.Allocs.pprof(from_c=false)