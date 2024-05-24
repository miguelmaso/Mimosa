using DrWatson
using Mimosa


function get_parameters()
 
  problemName = "static_plate"
  ptype = "Mechanics"
  regtype = "statics"
  meshfile = "cantilever.msh"

  # mechanical properties
  consmodel = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0, ρ=1.0)
 
  # boundary conditions 
  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  dirichletbc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolF(Λ) = Λ
  neu_F_tags = ["topcant"]
  neu_F_values = [[0.0, -0.1, 0.0]]
  neu_F_timesteps = [evolF]
  neumannbc = NeumannBC(neu_F_tags, neu_F_values, neu_F_timesteps)

  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 20
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  # Postprocessing
  is_vtk = true

  return @dict problemName ptype regtype meshfile consmodel dirichletbc neumannbc order solveropt is_vtk
end

main(; get_parameters()...)
