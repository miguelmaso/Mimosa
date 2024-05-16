using DrWatson
using Mimosa


function get_parameters(n_sec, sw, pot)

  problemName = "PlateBeam-$n_sec-Sec-Ord2_onoff_"
  for s in sw
    problemName = problemName*"_$s"
  end
  pot_10 = 10*pot
  problemName = problemName*"_ϕ$pot_10"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  meshfile = "PlateBeam4Sec.msh"


  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  modmec = NeoHookean3D(λ=10.0, μ=1.0)
  modelec = IdealDielectric(ε=1.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup_1"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolφ(Λ) = Λ
  earth_loc = Vector{String}()
  power_loc = Vector{String}()
  dir_φ_timesteps = Vector{typeof(evolφ)}()
  earth_val = []
  power_val = []
  for i in 1:n_sec
    append!(earth_val,0.0)
    append!(earth_loc, ["midsurf_$i"])
    push!(dir_φ_timesteps,evolφ)
    if sw[i]==0 #iseven(i)
      append!(power_loc,["bottomsurf_$i"])
      append!(power_val,pot)
      push!(dir_φ_timesteps,evolφ)
    elseif sw[i]==1
      append!(power_loc,["topsurf_$i"])
      append!(power_val,pot)
      push!(dir_φ_timesteps,evolφ)
    end
  end
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

n_sec = 4
sw = [0, 1, 1, 0] #0 = bottomsurf_   ;  1 = topsurf_; else = no potential in that section
pot = 0.1

main(; get_parameters(n_sec, sw, pot)...)

#  using PProf

#  PProf.Allocs.pprof(from_c=false)