using DrWatson
using Gridap

@testset "Static ThermoElectroMechanics" begin
 

  problemName = "test"
  ptype         = "ThermoElectroMechanics"
  soltype       = "monolithic"
  regtype       = "statics"
  meshfile = "TEMStaticSquare.msh"

  # thermal properties
  β = 2.233e-4
  e = 5209.0

  # coupling parameters
  f(θ::Float64)::Float64 = 1.0
  df(θ::Float64)::Float64 = 1.0

  # Constitutive models
  modmec = NeoHookean3D(λ=1e7, μ=1e6)
  modelec = IdealDielectric(ε=4.0)
  modterm = ThermalModel(Cv=100.0, θr=293.15, α=β * e, κ=10.0)
  consmodel = ThermoElectroMech(modterm, modelec, modmec, f, df)

  # boundary conditions 
  evolu(Λ)= 1.0
dir_u_tags = ["fixed"]
dir_u_values = [[0.0, 0.0, 0.0]]
dir_u_timesteps=[evolu]
Du=DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

evolφ(Λ)= Λ
dir_φ_tags = ["topsuf_3", "botsuf_1"]
dir_φ_values = [0.0, 5.0]
dir_φ_timesteps=[evolφ, evolφ]
Dφ=DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

evolθ(Λ)= Λ
dir_θ_tags = ["botsuf_1"]
dir_θ_values = [25.0]
dir_θ_timesteps=[evolθ]
Dθ=DirichletBC(dir_θ_tags, dir_θ_values, dir_θ_timesteps)

dirichletbc=MultiFieldBoundaryCondition([Du, Dφ, Dθ])
 
  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = false
  nr_iter = 20
  nr_ftol = 1e-5

  # Incremental solver
  nsteps = 5
  nbisec = 10

  # Postprocessing
  is_vtk = false

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

 params= @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk
 
 ph,cache = main(; params...)

 @test norm(get_free_dof_values(ph))==177.21808610230454
 @test cache.result.f_calls==3
 @test cache.result.residual_norm==6.359862254612381e-7

 end


 