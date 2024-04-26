using DrWatson
using Mimosa


function get_parameters()

  problemName = "TEM_StaticSquare"
  ptype = "ThermoElectroMechanics"
  couplingstrategy = "monolithic"
  model = "TEMStaticSquare.msh"

  # mechanical properties
  μ = 1e6
  λ = μ * 1e1

  # thermal properties
  β = 2.233e-4
  e = 5209.0
  θR = 293.15
  c0 = 100.0

  # electrical properties
  ε0 = 1
  εr = 4.0
  ε = εr * ε0

  # coupling parameters
  κ = 10.0
  f(θ::Float64)::Float64 = 1.0
  df(θ::Float64)::Float64 = 1.0

  # boundary conditions 

  dir_u_tags = ["fix"]
  dir_φ_tags = ["top", "bottom"]
  dir_θ_tags = ["bottom"]
  dir_u_values = [[1.0, 0.0, 0.0]]
  dir_φ_values = [0.0, 2.0e2]
  dir_θ_values = [θR]

  dir_tags=[dir_u_tags, dir_φ_tags, dir_θ_tags]
  dir_values=[dir_u_values, dir_φ_values, dir_θ_values]

  dirichletbc = @dict tags=dir_tags values=dir_values

  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-5

  nsteps = 5
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  return @dict problemName ptype couplingstrategy model μ λ β e θR c0 ε0 εr ε κ f df dirichletbc order solveropt
end
 
  main(; get_parameters()...)
 
 