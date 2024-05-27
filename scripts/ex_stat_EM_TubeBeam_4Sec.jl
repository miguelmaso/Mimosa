using DrWatson
using Mimosa


function get_parameters(pot)

  problemName = "TubeBeam4Sec2" 
  problemName = problemName*"_ϕ$pot"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  meshfile = "TubeBeam_SecT_8.msh"


  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  modmec = NeoHookean3D(λ=0, μ=0.03911e6)
  modelec = IdealDielectric(ε=8.8542e-12*4.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

  evolu(Λ) = 1.0
  dir_u_tags = ["fixed_surf_1-1", "fixed_surf_2-1", "fixed_surf_3-1", "fixed_surf_4-1", "fixed_surf_5-1", "fixed_surf_6-1", "fixed_surf_7-1", "fixed_surf_8-1"]
  dir_u_values = [[0.0,0.0,0.0] for i in dir_u_tags]
  dir_u_timesteps = [evolu for i in dir_u_tags]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolφ(Λ) = Λ
  earth_loc = ["int_surf_1-1", "int_surf_2-1", "int_surf_3-1", "int_surf_4-1", "int_surf_5-1", "int_surf_6-1", "int_surf_7-1", "int_surf_8-1", "int_surf_1-2", "int_surf_2-2", "int_surf_3-2", "int_surf_4-2", "int_surf_5-2", "int_surf_6-2", "int_surf_7-2", "int_surf_8-2", "int_surf_1-3", "int_surf_2-3", "int_surf_3-3", "int_surf_4-3", "int_surf_5-3", "int_surf_6-3", "int_surf_7-3", "int_surf_8-3", "int_surf_1-4", "int_surf_2-4", "int_surf_3-4", "int_surf_4-4", "int_surf_5-4", "int_surf_6-4", "int_surf_7-4", "int_surf_8-4"]
  power_loc = ["ext_surf_1-1","ext_surf_3-2","ext_surf_5-3","ext_surf_7-4"]
  earth_val = [0.0 for i in earth_loc]
  power_val = [pot for i in power_loc]
  dir_φ_tags = Vector{String}()
  append!(dir_φ_tags,earth_loc)
  append!(dir_φ_tags,power_loc)
  display(dir_φ_tags)
  dir_φ_timesteps = [evolφ for i in dir_φ_tags]
  display(dir_φ_timesteps)
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

pots = [5000.0]
for pot in pots
  ph, chache = main(; get_parameters(pot)...)
end


#  using PProf

#  PProf.Allocs.pprof(from_c=false)