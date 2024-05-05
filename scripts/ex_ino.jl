using DrWatson
using Mimosa


function get_parameters()

  problemName = "exampleINO"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  meshfile = "mesh_ino.msh"


  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  modmec = NeoHookean3D(λ=10.0, μ=1.0)
  modelec = IdealDielectric(ε=1.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

  evolu(Λ) = 1.0
  dir_u_tags = ["Fixed"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolφ(Λ) = Λ
  dir_φ_tags = ["Neumann"]
  dir_φ_values = [0.0]
  dir_φ_timesteps = [evolφ]
  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])

  # FE parameters
  order =2

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 1
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  # Postprocessing
  is_vtk = true

  return @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk
end

main(; get_parameters()...)

#  using PProf

#  PProf.Allocs.pprof(from_c=false)