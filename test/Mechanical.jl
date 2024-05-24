using DrWatson
using Gridap


@testset "Static Mechanical (Neumann)" begin
 
  problemName = "static_plate"
  ptype = "Mechanics"
  regtype = "statics"
  meshfile = "test_static_M.msh"

  # mechanical properties
  consmodel = MoneyRivlin3D(λ=3.0, μ1=1.0, μ2=0.0, ρ=1.0)
 
  # boundary conditions 
  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  dirichletbc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolF(Λ) = Λ
  dir_F_tags = ["topcant"]
  dir_F_values = [[0.0, -0.01, 0.0]]
  dir_F_timesteps = [evolF]
  neumannbc = NeumannBC(dir_F_tags, dir_F_values, dir_F_timesteps)

  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = false
  nr_iter = 20
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 20
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  # Postprocessing
  is_vtk = false

  params= @dict problemName ptype regtype meshfile consmodel dirichletbc neumannbc order solveropt is_vtk
 
ph,cache = main(; params...)

@test norm(get_free_dof_values(ph))  ==  321.2352673669686
@test cache.result.residual_norm  ==7.20725562564084e-14
end

@testset "Dynamic Mechanical (Dirichlet)" begin
 
    problemName = "test"
    ptype = "Mechanics"
    regtype = "dynamics"
    mesh_file = "test_static_EM.msh"
  
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
    nr_show_trace = false
    nr_iter = 20
    nr_ftol = 1e-12
  
    # Midpoint solver
    Δt = 5.0
    nsteps = 5
    αray = 0.4
    solveropt = @dict nr_show_trace nr_iter nr_ftol Δt nsteps αray
  
    # Postprocessing
    is_vtk = false
  
     params= @dict problemName ptype regtype mesh_file consmodel dirichletbc order solveropt is_vtk velocity
   
  ph, KE, EE,cache = main(; params...)

  @test norm(get_free_dof_values(ph)) ==3.20097810086495
  @test norm(KE) ==7.789152098340261e-5
  @test norm(EE) ==0.00013605028266911283
  @test cache.result.residual_norm ==5.683692991080569e-13

end


 