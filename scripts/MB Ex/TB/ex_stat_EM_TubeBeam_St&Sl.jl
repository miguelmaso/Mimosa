using DrWatson
using Mimosa
using Gridap.Visualization
using Gridap.TensorValues
using CSV
using DataFrames

t0 = time()

function get_parameters(pot, sw, St, Sl)

    problemName = "TubeBeam" 
    problemName = problemName*"_ϕ$pot"*"_St$St"*"_St$Sl"
    for s in sw
        problemName = problemName*"_$s"
    end
    ptype = "ElectroMechanics"
    soltype = "monolithic"
    regtype = "statics"
    meshfile = "TubeBeam_SecT_4-SecL_2.msh"


    # modmec = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0)
    modmec = NeoHookean3D(λ=0, μ=0.03911e6)
    modelec = IdealDielectric(ε=8.8542e-12*4.0)
    consmodel = ElectroMech(modmec, modelec)

    # Boundary conditions 

    evolu(Λ) = 1.0
    dir_u_tags = ["fixed_surf_1", "fixed_surf_2", "fixed_surf_3", "fixed_surf_4"]
    dir_u_values = [[0.0,0.0,0.0] for i in dir_u_tags]
    dir_u_timesteps = [evolu for i in dir_u_tags]
    Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

    evolφ(Λ) = Λ
    ext_surf_list = ["ext_surf_1", "ext_surf_2", "ext_surf_3", "ext_surf_4", "ext_surf_5", "ext_surf_6", "ext_surf_7", "ext_surf_8"]
    earth_loc = ["int_surf_1", "int_surf_2", "int_surf_3", "int_surf_4", "int_surf_5", "int_surf_6", "int_surf_7", "int_surf_8"]
    power_loc = []
    i = 1
    for io in sw
        if io == 1
            push!(power_loc,ext_surf_list[i])
        end
        i += 1
    end
    earth_val = [0.0 for i in earth_loc]
    power_val = [pot for i in power_loc]
    dir_φ_tags = Vector{String}()
    append!(dir_φ_tags,earth_loc)
    append!(dir_φ_tags,power_loc)
    dir_φ_timesteps = [evolφ for i in dir_φ_tags]
    dir_φ_values = []
    append!(dir_φ_values,earth_val)
    append!(dir_φ_values,power_val)
    
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
    is_vtk = false
    is_P_F = false

    return @dict problemName ptype soltype regtype meshfile consmodel dirichletbc order solveropt is_vtk is_P_F
end


#  Was not able to make it work this way
# function CenterLine(ph)
#     u = ph[1]
#     grid = u.cell_field.trian.grid.node_coordinates
#     node1 = grid[1]
#     node2 = grid[2]
#     r_yz1 = [node1[2],node1[3]]
#     r1 = norm(r_yz1)
#     r_yz2 = [node2[2],node2[3]]
#     r2 = norm(r_yz2)
#     r_mid = (r1 + r2)/2
#     mid_grid = []
#     tol = 0.001
#     r_mid_up = (1 + tol)*r_mid
#     r_mid_down = (1 - tol)*r_mid
#     for node in grid
#         r_yz = [node[2],node[3]]
#         r = norm(r_yz)
#         if r_mid_down<r<r_mid_up
#             push!(mid_grid,node)
#         end
#     end
#     # return mid_grid
#     plot(getproperty.(mid_grid,:data))
#     sort!(mid_grid, by = x -> x[1])
#     u_ = u.(mid_grid)
#     x_mid_grid = mid_grid + u_
#     plot(getproperty.(x_mid_grid,:data))
# end

function CenterLine(ph)

    u = ph[1]
    vd = visualization_data(u.cell_field.trian,"",cellfields=["u"=>u])
    vd = vd[1]
    X_grid = vd.grid.sub_grid.node_coordinates
    u_grid = vd.nodaldata["u"]
    x_grid = X_grid .+ u_grid
    sort_inidices = sortperm(X_grid, by = x -> x[1])
    x0 = X_grid[sort_inidices[1]][1]
    x_ = VectorValue(0.0,0.0,0.0)
    n = 0
    x_line = []
    tol = 0.00001
    for i in sort_inidices
        if (1-tol)*x0<=X_grid[i][1]<=(1+tol)*x0
            x_ += x_grid[i]
            n += 1
        else
            push!(x_line, x_./n)
            x0 = X_grid[i][1]
            x_ = x_grid[i]
            n = 1
        end
    end
    push!(x_line, x_./n)
    # plot(getproperty.(x_line,:data))
    return x_line
end

div = 4
pots = [2000.0]
conf_list = []
for i in [0,1], j in [0,1], k in [0,1], l in [0,1], m in [0,1], n in [0,1], o in [0,1], p in [0,1]
    push!(conf_list,[i,j,k,l,m,n,o,p])
end
St = 4
Sl = 2
ph_list = []
x_line_list = []
n = length(conf_list)
start = (div-1)*(n/4) + 1
finish = start + 1 # div*(n/4)
for i in Int(start):Int(finish)
    conf = conf_list[i]
    println(" ")
    println("!!!!!!!!    Configuration number $i / $n   !!!!!!!")
    println("!!!!!!!!        Configuration $conf        !!!!!!!")
    println(" ")
    ph, chache = main(; get_parameters(pots[1], conf, St, Sl)...)
    x_line = CenterLine(ph)
    push!(ph_list,ph)
    push!(x_line_list,x_line)
end

x_line_list = [getproperty.(x_line,:data) for x_line in x_line_list]
x_line_list = [x_line_list[i][j][k] for i in 1:lastindex(x_line_list), j in 1:lastindex(x_line_list[1]), k in 1:lastindex(x_line_list[1][1])]
df = DataFrame(vcat(x_line_list[:,:,1],x_line_list[:,:,2],x_line_list[:,:,3]), :auto)
CSV.write("data/csv/EM_TB_St4_Sl2_$div.csv", df)
t1 = time()
Δt = t1-t0
# test_read = CSV.File("data/csv/EM_TB_St4_Sl2_$div.csv") |> Tables.matrix