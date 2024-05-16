using DrWatson
using Mimosa


function get_parameters(pot)

  problemName = "TubeBeam" 
  problemName = problemName*"_ϕ$pot"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  meshfile = "TubeBeam.msh"


  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  modmec = NeoHookean3D(λ=0, μ=0.03911e6)
  modelec = IdealDielectric(ε=8.8542e-12*4.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

  evolu(Λ) = 1.0
  dir_u_tags = ["surf_2", "surf_7", "surf_12", "surf_17"]
  dir_u_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu, evolu, evolu, evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolφ(Λ) = Λ
  earth_loc = ["surf_5", "surf_10", "surf_15", "surf_20"]
  power_loc = ["surf_3"]
  dir_φ_timesteps = [evolφ, evolφ, evolφ, evolφ, evolφ]
  earth_val = [0.0, 0.0, 0.0, 0.0]
  power_val = [pot]
  display(dir_φ_timesteps)
  dir_φ_tags = Vector{String}()
  append!(dir_φ_tags,earth_loc)
  append!(dir_φ_tags,power_loc)
  display(dir_φ_tags)
  dir_φ_values = []
  append!(dir_φ_values,earth_val)
  append!(dir_φ_values,power_val)
  display(dir_φ_values)

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
  is_P_F = true

  return @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk is_P_F
end

pots = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
for pot in pots
  ph, chache = main(; get_parameters(pot)...)
end


#  using PProf

#  PProf.Allocs.pprof(from_c=false)