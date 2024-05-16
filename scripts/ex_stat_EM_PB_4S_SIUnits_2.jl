using DrWatson
using Mimosa
using Gridap
using Plots


function get_parameters(n_sec, sw, pot)

  problemName = "PB-S$n_sec-O2-PL" #PB = PlateBeam; S = # of sections; O2 = Order of elements; PL = Potential Location; ϕ = potential magnitude
  for s in sw
    problemName = problemName*"_$s"
  end
  pot_10 = 10*pot
  problemName = problemName*"_ϕ$pot_10"
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  meshfile = "PlateBeam4SecSI.msh"


  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  modmec = NeoHookean3D(λ=0, μ=0.03911e6)
  modelec = IdealDielectric(ε=8.8542e-12*4.0)
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

function plt_cl(ph,sw)
  uh = ph[1]
  t = 0.0:0.1e-3:100.0e-3
  X = [VectorValue(i,4.0e-3,0.4e-3) for i in t]
  uh_l = uh.(X)
  x = X + uh_l
  Z1 = [X[i][3] for i in 1:length(x)]
  X1 = [X[i][1] for i in 1:length(x)]
  z1 = [x[i][3] for i in 1:length(x)]
  x1 = [x[i][1] for i in 1:length(x)]
  return plot([x1,X1], [z1,Z1], title=sw, xticks=[0.0,0.05,0.1])
end

conf = []
for i in 0:2, j in 0:2, k in 0:2, l in 0:2
  push!(conf,[i,j,k,l])
end

n_sec = 4
pot = 2000.0
#sw = [0, 1, 1, 0] #0 = bottomsurf_   ;  1 = topsurf_; else = no potential in that section
uh_ = []
pl = []
for i in 26:54
  sw = conf[i]
  ph, chache = main(; get_parameters(n_sec, sw, pot)...)
  push!(uh_,ph[1])
  push!(pl,plt_cl(ph,sw))
end
plot(pl...,layout=(3,9))