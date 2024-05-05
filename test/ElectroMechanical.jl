using DrWatson
using Gridap

@testset "Static ElectroMechanics" begin

    problemName = "test"
    ptype = "ElectroMechanics"
    soltype = "monolithic"
    regtype = "statics"
    meshfile = "test_static_EM.msh"

    modmec = NeoHookean3D(λ=10.0, μ=1.0)
    modelec = IdealDielectric(ε=1.0)
    consmodel = ElectroMech(modmec, modelec)

    # Boundary conditions 
    evolu(Λ) = 1.0
    dir_u_tags = ["fixedup"]
    dir_u_values = [[0.0, 0.0, 0.0]]
    dir_u_timesteps = [evolu]
    Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

    evolφ(Λ) = Λ
    dir_φ_tags = ["midsuf", "topsuf"]
    dir_φ_values = [0.0, 0.1]
    dir_φ_timesteps = [evolφ, evolφ]
    Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

    dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])

    # FE parameters
    order = 1

    # NewtonRaphson parameters
    nr_show_trace = false
    nr_iter = 20
    nr_ftol = 1e-12

    # Incremental solver
    nsteps = 5
    nbisec = 10

    solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

    # Postprocessing
    is_vtk = false

    params = @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk

    ph,cache = main(; params...)

    @test norm(get_free_dof_values(ph))==45.909624929098094
    @test cache.result.f_calls==8
    @test cache.result.residual_norm==3.634744415170754e-14

end




@testset "Dynamic ElectroMechanics" begin

    problemName = "test"
    ptype = "ElectroMechanics"
    soltype = "monolithic"
    regtype = "dynamics"
    meshfile = "test_static_EM.msh"
  
    modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0, ρ=0.01)
    modelec = IdealDielectric(ε=1.0)
    consmodel = ElectroMech(modmec, modelec)
  
    # Boundary conditions 
  
    evolu(Λ) = 1.0
    dir_u_tags = ["fixedup"]
    dir_u_values = [[0.0, 0.0, 0.0]]
    dir_u_timesteps = [evolu]
    Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)
  
    evolφ(t) = min(t / 5.0, 1.0)
    dir_φ_tags = ["midsuf", "topsuf"]
    dir_φ_values = [0.0, 0.2]
    dir_φ_timesteps = [evolφ, evolφ]
    Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)
  
    dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])
  
    # initial conditions
    velocity(x) = VectorValue(0.0, 0.0, 0.0)
  
    # FE parameters
    order = 1
  
    # NewtonRaphson parameters
    nr_show_trace = false
    nr_iter = 20
    nr_ftol = 1e-12
  
    # Midpoint solver
    Δt = 0.5
    nsteps = 5
    solveropt = @dict nr_show_trace nr_iter nr_ftol Δt nsteps
  
    # Postprocessing
    is_vtk = false
  
   params= @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk velocity
   
   ph, KE, EE, cache = main(; params...)

   @test norm(get_free_dof_values(ph))==0.9831056309907463 
   @test norm(KE)==0.0020194079090447422
   @test norm(EE)==5.05365782228524
   @test cache.result.residual_norm==2.267717263970681e-15 

end