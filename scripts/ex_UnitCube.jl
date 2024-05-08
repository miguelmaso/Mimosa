using DrWatson
using Mimosa


function get_parameters()

  problemName = "Unit_Cube"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  meshfile = "Unit_Cube.msh"


  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  modmec = NeoHookean3D(λ=10.0, μ=1.0)
  modelec = IdealDielectric(ε=1.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

  evolu(Λ) = Λ
  dir_u_tags = ["face_1", "face_26"] #, "face_17", "face_21", "face_25", "face_26"]
  # Ux(x) = 1.1*x #[[0.0, 0.0, 0.0]]
  dir_u_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]] # , [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]# [Ux, Ux, Ux, Ux, Ux, Ux]
  dir_u_timesteps = [evolu, evolu] # , evolu, evolu, evolu, evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolφ(Λ) = 1
  dir_φ_tags = ["face_1"]
  dir_φ_values = [0.0]
  dir_φ_timesteps = [evolφ]
  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])

  # FE parameters
  order = 2

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 5
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  # Postprocessing
  is_vtk = true

  return @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk
end

main(; get_parameters()...)