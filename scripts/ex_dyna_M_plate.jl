using DrWatson
using Mimosa
using Gridap
using Plots

function get_parameters()

  problemName = "Dynatest"
  ptype = "Mechanics"
  regtype = "dynamics"
  mesh_file = "ex2_mesh.msh"

  # mechanical properties
  consmodel = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0, ρ=1.0)

  # boundary conditions 
  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  dirichletbc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  # initial conditions
  velocity(x) = VectorValue(0.0, x[3] * 0.1 / 40.0, 0.0)

  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  # Midpoint solver
  Δt = 5.0
  nsteps = 500
  solveropt = @dict nr_show_trace nr_iter nr_ftol Δt nsteps

  # Postprocessing
  is_vtk = true

  return @dict problemName ptype regtype mesh_file consmodel dirichletbc order solveropt is_vtk velocity
end

ph, KE, EE = main(; get_parameters()...)

plot(KE, label="Kinetic Energy")
plot!(EE, label="Elastic Energy")
plot!(KE + EE, label="Total Energy")
