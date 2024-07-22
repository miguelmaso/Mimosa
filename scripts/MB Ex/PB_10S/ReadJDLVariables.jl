using Gridap
using JLD2
using Gridap.Visualization

uh_list = []
for i in 1:6
    jldopen("data/JLD2_Fields/PB_10S/uh_EM_PB_10S_$i.jld2","r") do f
        push!(uh_list,read(f, "uh_"))
    end
end
jldopen("data/JLD2_Fields/PB_10S/uh_EM_PB_10S_1x.jld2","r") do f
    push!(uh_list,read(f, "uh_"))
end
uh_list = reduce(append!,uh_list)
y_mid = (3.0/7.0)*0.008
z_mid = 0.0004
y_list = []
x_list = []
for uh in uh_list
    vd = visualization_data(uh.cell_field.trian,"",cellfields=["u"=>uh])
    vd = vd[1]
    X_grid = vd.grid.sub_grid.node_coordinates
    u_grid = vd.nodaldata["u"]
    x_grid = X_grid .+ u_grid
    sort_inidices = sortperm(X_grid, by = x -> x[1])
    x0 = X_grid[sort_inidices[1]][1]
    y = []
    x = []
    tol = 0.00001
    n = true
    for i in sort_inidices
        if (1-tol)*x0<=X_grid[i][1]<=(1+tol)*x0 && n
            if (1-tol)*y_mid<=X_grid[i][2]<=(1+tol)*y_mid && (1-tol)*z_mid<=X_grid[i][3]<=(1+tol)*z_mid
                push!(y,x_grid[i][3])
                push!(x,X_grid[i][1])
                n = false
            end
        else
            if (1-tol)*y_mid<=X_grid[i][2]<=(1+tol)*y_mid && (1-tol)*z_mid<=X_grid[i][3]<=(1+tol)*z_mid
                push!(y,x_grid[i][3])
                push!(x,X_grid[i][1])
                n = false
            end
            x0 = X_grid[i][1]
            n = true
        end
    end
    push!(y_list,y)
    push!(x_list,x)
end