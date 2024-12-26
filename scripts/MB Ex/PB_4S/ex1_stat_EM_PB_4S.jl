using DrWatson
using Mimosa
using Gridap
using Plots
using CSV
using DataFrames
using Tables

function get_parameters(n_sec, conf, pot,csv_funct)
  # Transform binary
  sw = Array{Int}(undef, (4))
  for i in 1:n_sec
    for j in 0:3
        if conf[[(((i-1)*2)+1):(((i-1)*2)+2)...]] == digits(j,base=2,pad=2)
          sw[i] = j
          break
        end
    end
  end
  
  
  ptype = "ElectroMechanics"
  soltype = "monolithic"
  regtype = "statics"
  # diffstrat = "analytic"
  diffstrat = "autodiff"
  # meshfile = "PlateBeam4SecSI.msh"
  meshfile = "PlateBeame4S_BC.msh"
  problemName = "PB-S$n_sec-O2-/$diffstrat/Yeoh/PL" #PB = PlateBeam; S = # of sections; O2 = Order of elements; PL = Potential Location; ϕ = potential magnitude
  for s in conf
    problemName = problemName*"_$s"
  end
  pot_10 = 10*pot
  problemName = problemName*"_ϕ$pot_10"


  # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
  # modmec = NeoHookean3D(λ=0, μ=0.03911e6)
  modmec = Yeoh(C₁ = 0.0693e6, C₂ = -8.88e2, C₃ = 16.7, κ = 0.0693e8)
  # modmec = NeoHookean3DNearlyIncomp(λ=0.03911e6*1e3, μ=0.03911e6)
  # modmec = NeoHookean3D(λ=1e5, μ=0.03911e6)
  modelec = IdealDielectric(ε=8.8542e-12*4.0)
  consmodel = ElectroMech(modmec, modelec)

  # Boundary conditions 

  evolu(Λ) = 1.0
  dir_u_tags = ["point_zy","point_z","fixedup_1"]
  dir_u_values = [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]
  masks = [(true,true,true),(true,false,true),(true,false,false)]
  dir_u_timesteps = [evolu,evolu,evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps,masks)
  # println(Du)

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
    if sw[i]==1 #iseven(i)
      append!(power_loc,["bottomsurf_$i"])
      append!(power_val,pot)
      push!(dir_φ_timesteps,evolφ)
    elseif sw[i]==2
      append!(power_loc,["topsurf_$i"])
      append!(power_val,pot)
      push!(dir_φ_timesteps,evolφ)
    elseif sw[i]==3 #iseven(i)
      append!(power_loc,["bottomsurf_$i"])
      append!(power_val,pot)
      push!(dir_φ_timesteps,evolφ)
      append!(power_loc,["topsurf_$i"])
      append!(power_val,pot)
      push!(dir_φ_timesteps,evolφ)
    end
  end
  # display(dir_φ_timesteps)
  dir_φ_tags = Vector{String}()
  append!(dir_φ_tags,earth_loc)
  append!(dir_φ_tags,power_loc)
  # display(dir_φ_tags)
  dir_φ_values = []
  append!(dir_φ_values,earth_val)
  append!(dir_φ_values,power_val)
  # display(dir_φ_values)

  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  dirichletbc = MultiFieldBoundaryCondition([Du, Dφ])

  # FE parameters
  order = 2

  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 15
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 5
  nbisec = 10

  solveropt = @dict nr_show_trace nr_iter nr_ftol nsteps nbisec

  # Postprocessing
  is_vtk = true
  is_P_F = true

  init_sol_bool, U_ap, X_ap = false, nothing, nothing

  csv_bool = true
  csv_funct_ = csv_funct

  return @dict problemName ptype soltype regtype diffstrat meshfile consmodel dirichletbc order solveropt is_vtk is_P_F init_sol_bool U_ap X_ap csv_bool csv_funct_
end

function plt_cl(ph,sw)
  uh = ph[1]
  t = 0.0:1e-3:100.0e-3
  X = [VectorValue(i,4.0e-3,0.4e-3) for i in t]
  uh_l = uh.(X)
  x = X + uh_l
  Z1 = [X[i][3] for i in 1:lastindex(x)]
  X1 = [X[i][1] for i in 1:lastindex(x)]
  z1 = [x[i][3] for i in 1:lastindex(x)]
  x1 = [x[i][1] for i in 1:lastindex(x)]
  pl_ = plot([x1,X1], [z1,Z1], title=sw, xticks=[0.0,0.05,0.1])
  return pl_, z1
end

function csv_cl(ph)
  uh = ph[1]
  t = 0.0:1e-3:100.0e-3
  X = [VectorValue(i,4.0e-3,0.4e-3) for i in t]
  uh_l = uh.(X)
  x = X + uh_l
  # Z1 = [X[i][3] for i in 1:lastindex(x)]
  # X1 = [X[i][1] for i in 1:lastindex(x)]
  z1 = [x[i][3] for i in 1:lastindex(x)]
  x1 = [x[i][1] for i in 1:lastindex(x)]
  df = DataFrame(x_ = vcat(z1,x1))
  return df
end

function run()
  Conf = []
  for i1 in 0:1, i2 in 0:1, i3 in 0:1, i4 in 0:1, i5 in 0:1, i6 in 0:1, i7 in 0:1, i8 in 0:1
    push!(Conf,[i1,i2,i3,i4,i5,i6,i7,i8])
  end

  n_sec = 4
  pot = 5000.0
  for i in 154:154
    conf = Conf[i]
    ph, chache = main(; get_parameters(n_sec, conf, pot,csv_cl)...)
  end
end

function run(start,finish)
  Conf = []
  for i1 in 0:1, i2 in 0:1, i3 in 0:1, i4 in 0:1, i5 in 0:1, i6 in 0:1, i7 in 0:1, i8 in 0:1
    push!(Conf,[i1,i2,i3,i4,i5,i6,i7,i8])
  end

  n_sec = 4
  pot = 5000.0
  for i in start:finish
    conf = Conf[i]
    ph, chache = main(; get_parameters(n_sec, conf, pot,csv_cl)...)
  end
end