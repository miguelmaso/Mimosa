using DrWatson
using Mimosa


function get_parameters()

  problemName = "M_Plate"
  ptype = "Mechanics"
  model = "ex2_mesh.msh"

  # mechanical properties
  μ = 1.0
  λ = 10.0
 
 
  # boundary conditions 
  dir_tags = ["fixedup"]
  dir_values = [[0.0, 0.0, 0.0]]
  neu_tags = ["topsuf"]
  neu_values = [[0.0, 0.0, -1.0]]
 
 
  dirichletbc = @dict tags=dir_tags values=dir_values
  neumannbc   = @dict tags=neu_tags values=neu_values

   # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  nsteps = 5
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  return @dict problemName ptype  model μ λ dirichletbc neumannbc order solveropt
end

main(; get_parameters()...)
  