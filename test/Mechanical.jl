using DrWatson
using Gridap

@testset "Dynamic Mechanical" begin
 
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
    solveropt = @dict nr_show_trace nr_iter nr_ftol Δt nsteps
  
    # Postprocessing
    is_vtk = false
  
     params= @dict problemName ptype regtype mesh_file consmodel dirichletbc order solveropt is_vtk velocity
   
  ph, KE, EE,cache = main(; params...)

  @test norm(get_free_dof_values(ph))==31.981666176806325 
  @test norm(KE)==0.9501834649502897
  @test norm(EE)==0.004681828202450063
  @test cache.result.residual_norm==2.402538888321426e-13 
end


 