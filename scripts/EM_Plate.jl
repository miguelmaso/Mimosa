using DrWatson
using Mimosa


function get_parameters()

  problemName = "EMPlate"
  ptype = "ElectroMechanics"
  couplingstrategy = "monolithic"
  model = "ex2_mesh.msh"

  # mechanical properties
  μ = 1.0
  λ = 10.0
 
  # electrical properties
  ε0 = 1.0
  εr = 1.0
  ε = εr * ε0
 
  # boundary conditions 
  dir_u_tags = ["fixedup"]
  dir_φ_tags = ["midsuf", "topsuf"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_φ_values = [0.0, 0.01]

    
  dir_tags=[dir_u_tags, dir_φ_tags]
  dir_values=[dir_u_values, dir_φ_values]

  dirichletbc = @dict tags=dir_tags values=dir_values
 
   # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  nsteps = 5
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  return @dict problemName ptype couplingstrategy model μ λ ε0 εr ε dirichletbc order solveropt
end

main(; get_parameters()...)
 
#  using PProf

#  PProf.Allocs.pprof(from_c=false)