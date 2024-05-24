using DrWatson
using Mimosa


function main(pname = "test_")
    
  # Problem setting
  ptype = "Mechanics"
  regtype = "statics"
  print_heading(@dict ptype regtype pname)

  is_vtk = true
  simdir_ = datadir("sims", pname)
  setupfolder(simdir_)

  # Constitutive models
  PhysModel = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0, ρ=1.0)

  # Derivatives
  Ψ, ∂Ψu, ∂Ψuu = PhysModel(DerivativeStrategy{:analytic}())
  DΨ = @ntuple Ψ ∂Ψu ∂Ψuu

  # grid model
  meshfile = "cantilever.msh"
  model = GmshDiscreteModel(datadir("models", mesh_file))
  if is_vtk 
      writevtk(model, simdir_ * "/DiscreteModel")
  end

  # Setup integration
  order = 1
  degree = 2 * order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  
  # boundary conditions 
  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  dirichletbc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolF(Λ) = Λ
  neu_F_tags = ["topcant"]
  neu_F_values = [[0.0, -0.1, 0.0]]
  neu_F_timesteps = [evolF]
  neumannbc = NeumannBC(neu_F_tags, neu_F_values, neu_F_timesteps)

  dΓ=get_Neumann_dΓ(model,neumannbc,degree)


  #  FE spaces
  fe_spaces = get_FE_spaces(MechanicalProblem{:statics}(), model, order, dirichletbc)

  # WeakForms
  res(u, v) = residual(Mechano, u,v,∂Ψu,dΩ)
  jac(u, du, v) = jacobian(Mechano, u, du, v, ∂Ψuu, dΩ)


  @timeit pname begin
      println("Defining Nonlinear solver")
      # NewtonRaphson solver
      solveropt = _get_kwarg(:solveropt, kwargs)
      nlsolver = get_FE_solver(solveropt)

      # Initialization
      x0 = zeros(Float64, num_free_dofs(fe_spaces.V))
      ph = FEFunction(fe_spaces.U, x0)
      @show size(get_free_dof_values(ph))


      post_params = @dict Ω is_vtk simdir_

      solver_params = @dict fe_spaces dirichletbc neumannbc Ω dΩ dΓ DΨ res jac solveropt nlsolver post_params

      ph,cache  = IncrementalSolver(problem, ph, solver_params)

   end


end



function get_parameters()
 
  problemName = "static_plate"
  ptype = "Mechanics"
  regtype = "statics"
  meshfile = "cantilever.msh"

  # mechanical properties
  consmodel = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0, ρ=1.0)
 
  # boundary conditions 
  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  dirichletbc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolF(Λ) = Λ
  neu_F_tags = ["topcant"]
  neu_F_values = [[0.0, -0.1, 0.0]]
  neu_F_timesteps = [evolF]
  neumannbc = NeumannBC(neu_F_tags, neu_F_values, neu_F_timesteps)

  # FE parameters
  order = 1

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 20
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  # Postprocessing
  is_vtk = true

  return @dict problemName ptype regtype meshfile consmodel dirichletbc neumannbc order solveropt is_vtk
end

main(; get_parameters()...)
