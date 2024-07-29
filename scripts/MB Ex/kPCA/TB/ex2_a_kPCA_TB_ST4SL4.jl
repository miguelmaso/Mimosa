using CSV
using Tables
using LinearAlgebra
using Plots
using NearestNeighbors
using Optim

function ReadData2(pot,parts)
    j = pot
    i = 0
    _X = CSV.File("data/csv/EM_TB_St4_Sl4_Phi$(j)_Random/EM_TB_St4_Sl4_$i.csv") |> Tables.matrix
    _X = _X'
    n = 64
    X_ = []
    push!(X_,_X[:,[1:n...]])
    push!(X_,_X[:,[n+1:2*n...]])
    push!(X_,_X[:,[2*n+1:3*n...]])
    # X_[1] = hcat(X_[1],_X[:,[1:n...]])
    # X_[2] = hcat(X_[2],_X[:,[n+1:2*n...]])
    # X_[3] = hcat(X_[3],_X[:,[2*n+1:3*n...]])
    for i in 1:parts
        _X = CSV.File("data/csv/EM_TB_St4_Sl4_Phi$(j)_Random/EM_TB_St4_Sl4_$i.csv") |> Tables.matrix
        _X = _X'
        n = 50
        # X_ = []
        # push!(X_,_X[:,[1:n...]])
        # push!(X_,_X[:,[n+1:2*n...]])
        # push!(X_,_X[:,[2*n+1:3*n...]])
        X_[1] = hcat(X_[1],_X[:,[1:n...]])
        X_[2] = hcat(X_[2],_X[:,[n+1:2*n...]])
        X_[3] = hcat(X_[3],_X[:,[2*n+1:3*n...]])
    end
    X = reduce(vcat,X_)
    return X, X_
end

function kPOD(Κ,X,k)
    m, n = size(X)
    # X̄ = [1. for i in 1:n]
    # mean!(X̄',X)
    # for i in 1:n
    #     X[:,i] = X[:,i].-X̄[i]
    # end
    G = [Κ(X[:,i],X[:,j]) for i in 1:n, j in 1:n]
    # display(G)
    II = ones(n,n)
    Ḡ = G - (1/n)*G*II - (1/n)*II*G + (1/n^2)*II*G*II 
    # display(Ḡ)
    f(λ)=-abs(λ)
    Λ_ = eigen(Ḡ,sortby=f)
    Λ = Λ_.values
    U = Λ_.vectors
    Σ = diagm(real.(sqrt.(complex(Λ))))
    Σ_ = Σ[:,[1:k...]]
    U_ = U*pinv(Σ_')
    # U_ = U[:,1:k]
    return Λ, U, U_, Ḡ, G
end

function plotSet(idxs,conf_list,X_)
    idxs_ = copy(idxs)
    y_min = minimum(X_[2])
    y_max = maximum(X_[2])
    z_min = minimum(X_[3])
    z_max = maximum(X_[3])
    conf = idxs_[1]
    Cc = conf_list[conf]
    p = plot(X_[1][:,conf],X_[2][:,conf],X_[3][:,conf], xlims =(0.0,0.11),
    ylims=(y_min,y_max), zlims=(z_min,z_max),size=(600,800),
    label="Conf:$conf : $Cc", linewidth=6, xlabel="x", ylabel="y",
    zlabel="z", camera = (45, 45), legend=:outerbottom, legend_columns=2)
    popfirst!(idxs_)
    for idx in idxs_
        conf = idx
        Cc = conf_list[conf]
        p = plot!(X_[1][:,conf],X_[2][:,conf],X_[3][:,conf],
        label="Conf:$conf : $Cc", linewidth=6)
    end
    display(p)
end

function isomap1(neighbors,Z_)
    n = length(eachcol(Z_))
    D_G = [[] for i in 1:n]
    count = []
    count2 = []
    Threads.@threads for i in 1:n
        d_G, prev_ = Dijkstra(Z_,i,neighbors)
        D_G[i] = d_G
        push!(count,i)
        ma = maximum(count)
        push!(count2,ma)
        mi = minimum(count2)
        per = round(100*((ma-mi)/(n-mi)))
        print("\r$ma - $mi - %$per")
    end
    print("\n")
    D_G = reduce(hcat,D_G)
    D_G_sym = 0.5*(D_G+D_G')
    D_G_sq = D_G_sym.^2
    C = I - (1/n)*ones(n,n)
    B = -0.5*C*D_G_sq*C
    f(λ) = -real(λ)
    _iso = eigen(B, sortby=f)
    Λ_iso = real.(_iso.values)
    U_iso = real.(_iso.vectors)
    Λ_iso_m = diagm(sqrt.(Λ_iso[[1,2]]))
    U_iso_m = U_iso[:,[1,2]]
    X_iso = U_iso_m*Λ_iso_m
    return X_iso', D_G_sym
end

function Dijkstra(data,i,nieghbors)
    m, n = size(data)
    dist = [Inf64 for i in 1:n]
    prev = [0 for i in 1:n]
    nodes = data
    kdtree = KDTree(nodes; leafsize = 10)
    dist[i] = 0
    visited = []
    dist_ = copy(dist)
    while length(visited)<n
        # println(length(visited))
        u = argmin(dist_)
        append!(visited,u)
        distu = dist[u]
        idxs, dists = knn(kdtree, data[:,u], nieghbors, true,
        x -> maximum(isequal.(x,visited)))
        for i in 1:lastindex(idxs)
            alt = distu + dists[i]
            if alt<dist[idxs[i]]
                dist[idxs[i]]=alt
                prev[idxs[i]]=u
            end
        end
        dist_ = copy(dist)
        dist_[visited].=Inf64
    end
    return dist, prev
end

function VectorSearch(Z_,conf,list)
    n = length(list)
    z = Z_[:,list]
    v = [z[:,i]-Z_[:,1] for i in 1:n]
    z = Z_[:,1]
    for i in 1:Int(lastindex(conf)/4)
        for j in 1:16
            if conf[[(((i-1)*4)+1):(((i-1)*4)+4)...]] == digits(j-1,base=2,pad=4)
                z = z + v[((i-1)*16)+j]
            else
                z = z
            end
        end
    end
    return z
end

function VectorSearch(Z_,conf,list,∇zᵤ)
    n = length(list)
    z = Z_[:,list]
    v = [z[:,i]-Z_[:,1] for i in 1:n]
    z = Z_[:,1]
    R = I
    u_0 = ∇zᵤ(z[1],z[2])
    for i in 1:lastindex(conf)
        # v_ = v[i]
        # v_[1] = conf[i]*v_[1]
        if conf[i] == 0
            z = z
        elseif conf[i]==1
            z = z + R*v[i]
        end
        # println(R)
        u_1 = ∇zᵤ(z[1],z[2])
        R = RotM(u_1,u_0)
        # u_0 = u_1
    end
    return z
end


function RotM(v1,v2)
    v = cross(v1,v2)
    vₓ = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
    c = dot(v1,v2)
    R = I + vₓ + (1/(1+c))*vₓ*vₓ
    return R
end

function Objective(β)
    println("β = $β")
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Λ, U, U_, Ḡ, G = kPOD(Κ, X, 3)
    Z_ = real.(U_'*Ḡ)
    neighbors = 25
    Y_, _ = isomap1(neighbors,Z_)
    VS_Conf_list = [1:64...]
    Y_gen = []
    for i in 1:lastindex(eachcol(conf_list))
        conf = conf_list[:,i]
        push!(Y_gen,VectorSearch(Y_,conf,VS_Conf_list))
    end
    Y_gen = reduce(hcat,Y_gen)
    Err_TS = norm(Y_gen-Y_)/norm(Y_)
    println("Error in TS = $Err_TS")
    return Err_TS
end

function plot_x(x,L)
    n = length(x)
    x_ = x[[1:Int(n/3)...]]
    y_ = x[[Int(n/3)+1:Int(n/3)*2...]]
    z_ = x[[(Int(n/3)*2)+1:n...]]
    p = plot(x_,y_,z_,xlabel = "x", ylabel = "y", zlabel = "z",label=L)
    display(p)
end

function plot_x(x)
    n = length(x)
    x_ = x[[1:Int(n/3)...]]
    y_ = x[[Int(n/3)+1:Int(n/3)*2...]]
    z_ = x[[(Int(n/3)*2)+1:n...]]
    p = plot(x_,y_,z_,xlabel = "x", ylabel = "y", zlabel = "z")
    display(p)
end

function plot_x!(x,L)
    n = length(x)
    x_ = x[[1:Int(n/3)...]]
    y_ = x[[Int(n/3)+1:Int(n/3)*2...]]
    z_ = x[[(Int(n/3)*2)+1:n...]]
    p = plot!(x_,y_,z_,xlabel = "x", ylabel = "y", zlabel = "z",label=L)
    display(p)
end


function plot_x!(x)
    n = length(x)
    x_ = x[[1:Int(n/3)...]]
    y_ = x[[Int(n/3)+1:Int(n/3)*2...]]
    z_ = x[[(Int(n/3)*2)+1:n...]]
    p = plot!(x_,y_,z_,xlabel = "x", ylabel = "y", zlabel = "z")
    display(p)
end

function ReverseMap(X,Z_,z,n)
    w = [1/sqrt(dot(z-zi,z-zi)) for zi in eachcol(Z_)]
    # w = [exp(-sqrt(dot(z-zi,z-zi))) for zi in eachcol(Z_)]
    # display(w)
    w_sort = sortperm(w, rev=true)
    w_ns = [w[w_sort[1]]]
    X_ns = X[:,w_sort[1]]
    Z_ns = Z_[:,w_sort[1]]
    for i in 2:n
        push!(w_ns,w[w_sort[i]])
        X_ns = hcat(X_ns,X[:,w_sort[i]])
        Z_ns = hcat(Z_ns,Z_[:,w_sort[i]])
    end
    # w_ns = w_ns/sum(w_ns)
    w_ns = pinv(Z_ns)*z
    w_ns = w_ns/sum(w_ns)
    x = X_ns*w_ns
    return x, w_ns, Z_ns
end

function bin_to_int16(bin)
    num = 0
    for i in 1:16
        if bin == digits(i-1,base=2,pad=4)
            num = i
        end
    end
    return num
end

function RotM(v1,v2)
    v = cross(v1,v2)
    vₓ = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
    c = dot(v1,v2)
    R = I + vₓ + (1/(1+c))*vₓ*vₓ
    return R
end

function ReverseMap2(y_gen,conf_gen,Y_,X,conf,nh)
    n_ = Int(length(conf_gen)/4)
    n__ = lastindex(eachcol(X))
    x_gen = []
    for j in 1:n_
        param1 = []
        for i in 1:n__
            push!(param1,bin_to_int16(conf[[(((j-1)*4)+1):(((j-1)*4)+4)...],i]))
        end
        # println(param1)
        sort_param1 = sortperm(param1)
        group_param = 1
        group = [[] for i in 1:16]
        count = 1
        for i in 1:n__
            if param1[sort_param1[i]]==group_param
                push!(group[count],sort_param1[i])
            else
                count += 1
                group_param += 1
                push!(group[count],sort_param1[i])
            end
        end
        # println(group)
        param1_test = bin_to_int16(conf_gen[[(((j-1)*4)+1):(((j-1)*4)+4)...]])
        # println("")
        # println(param1_test)
        # println("")
        sec = [((7*(j-1)+1)):((7*j)+1)...]
        # if j==10
        #     append!(sec,101)
        # end
        # println((X[:,group[param1_test]],
        # Y_[:,group[param1_test]],y_gen,nh))
        x_gen_, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]],
        Y_[:,group[param1_test]],y_gen,nh)
        n = length(x_gen_)
        # println(x_gen_)
        x_gen_j = []
        push!(x_gen_j,x_gen_[[1:Int(n/3)...]][sec])
        push!(x_gen_j,x_gen_[[Int(n/3)+1:Int(n/3)*2...]][sec])
        push!(x_gen_j,x_gen_[[(Int(n/3)*2)+1:n...]][sec])
        push!(x_gen,x_gen_j)
    end
    for i in 2:n_
        for j in 1:3
            Δ = x_gen[i][j][begin]-x_gen[i-1][j][end]
            x_gen[i][j] = x_gen[i][j].-Δ
        end
        # println(i)
        x_ = reduce(hcat,x_gen[i])
        x_ = x_'
        b = x_[:,1]
        x_prev = reduce(hcat,x_gen[i-1])
        x_prev = x_prev'
        v2 = (x_prev[:,end]-x_prev[:,end-1])./norm(x_prev[:,end]-x_prev[:,end-1])
        v1 = (x_[:,begin+1]-x_[:,begin])./norm(x_[:,begin+1]-x_[:,begin])
        rot = RotM(v1,v2)
        # println(v1)
        # println(v2)
        # println(rot)
        B = reduce(hcat,[b for i in 1:lastindex(eachcol(x_))])
        x_ = x_-B
        x_ = rot*x_
        x_ = x_+B
        for j in 1:3
            x_gen[i][j] = x_[j,:]
        end
        # for k in 1:1
        #     println(k)
        #     # rotation in direction z
        #     θ1 = abs(atan(((x_gen[i-1][1][end] - x_gen[i-1][1][end-1])/(x_gen[i-1][2][end] - x_gen[i-1][2][end-1]))))
        #     θ2 = abs(atan(((x_gen[i][1][begin+1] - x_gen[i][1][begin])/(x_gen[i][2][begin+1] - x_gen[i][2][begin]))))
        #     Δθ = (θ1 - θ2)
        #     println("z")
        #     println((θ1*(180/pi)))
        #     println((θ2*(180/pi)))
        #     println((Δθ*(180/pi)))
        #     rot = [cos(Δθ) -sin(Δθ) 0.0; sin(Δθ) cos(Δθ) 0.0; 0.0 0.0 1.0]
        #     x_ = reduce(hcat,x_gen[i])
        #     x_ = x_'
        #     b = x_[:,1]
        #     B = reduce(hcat,[b for i in 1:lastindex(eachcol(x_))])
        #     x_ = x_-B
        #     x_ = rot*x_
        #     x_ = x_+B
        #     for j in 1:3
        #         x_gen[i][j] = x_[j,:]
        #     end
        #     # rotation in direction y
        #     θ1 = abs(atan(1/((x_gen[i-1][3][end] - x_gen[i-1][3][end-1])/(x_gen[i-1][1][end] - x_gen[i-1][1][end-1]))))
        #     θ2 = abs(atan(1/((x_gen[i][3][begin+1] - x_gen[i][3][begin])/(x_gen[i][1][begin+1] - x_gen[i][1][begin]))))
        #     Δθ = (θ1 - θ2)
        #     println("y")
        #     println((θ1*(180/pi)))
        #     println((θ2*(180/pi)))
        #     println((Δθ*(180/pi)))
        #     # println((Δθ*(180/pi)))
        #     rot = [cos(Δθ) 0.0 -sin(Δθ); 0.0 1.0 0.0; sin(Δθ) 0.0 cos(Δθ)]
        #     x_ = reduce(hcat,x_gen[i])
        #     x_ = x_'
        #     b = x_[:,1]
        #     B = reduce(hcat,[b for i in 1:lastindex(eachcol(x_))])
        #     x_ = x_-B
        #     x_ = rot*x_
        #     x_ = x_+B
        #     for j in 1:3
        #         x_gen[i][j] = x_[j,:]
        #     end
        #     # rotation in direction x
        #     θ1 = abs(atan(1/((x_gen[i-1][2][end] - x_gen[i-1][2][end-1])/(x_gen[i-1][3][end] - x_gen[i-1][3][end-1]))))
        #     θ2 = abs(atan(1/((x_gen[i][2][begin+1] - x_gen[i][2][begin])/(x_gen[i][3][begin+1] - x_gen[i][3][begin]))))
        #     Δθ = (θ1 - θ2)
        #     println("x")
        #     println((θ1*(180/pi)))
        #     println((θ2*(180/pi)))
        #     println((Δθ*(180/pi)))
        #     # println((Δθ*(180/pi)))
        #     rot = [1.0 0.0 0.0; 0.0 cos(Δθ) -sin(Δθ); 0.0 sin(Δθ) cos(Δθ)]
        #     x_ = reduce(hcat,x_gen[i])
        #     x_ = x_'
        #     b = x_[:,1]
        #     B = reduce(hcat,[b for i in 1:lastindex(eachcol(x_))])
        #     x_ = x_-B
        #     x_ = rot*x_
        #     x_ = x_+B
        #     print()
        #     for j in 1:3
        #         x_gen[i][j] = x_[j,:]
        #     end
        # end
    end



    x_gen_ = []
    for i in 1:3, j in 1:n_
        if j==n_
            append!(x_gen_,x_gen[j][i])
        else
            pop!(x_gen[j][i])
            append!(x_gen_,x_gen[j][i])
        end
    end
    return x_gen_
end

function ReverseMap3(y_gen,conf_gen,Y_,X,conf,nh)
    n_ = Int(length(conf_gen)/4)
    n__ = lastindex(eachcol(X))
    x_gen = []
    for j in 1:n_
        param1 = []
        for i in 1:n__
            push!(param1,bin_to_int16(conf[[(((j-1)*4)+1):(((j-1)*4)+4)...],i]))
        end
        # println(param1)
        sort_param1 = sortperm(param1)
        group_param = 1
        group = [[] for i in 1:16]
        count = 1
        for i in 1:n__
            if param1[sort_param1[i]]==group_param
                push!(group[count],sort_param1[i])
            else
                count += 1
                group_param += 1
                push!(group[count],sort_param1[i])
            end
        end
        # println(group)
        param1_test = bin_to_int16(conf_gen[[(((j-1)*4)+1):(((j-1)*4)+4)...]])
        # println("")
        # println(param1_test)
        # println("")
        sec = [((7*(j-1)+1)):((7*j)+1)...]
        # if j==10
        #     append!(sec,101)
        # end
        # println((X[:,group[param1_test]],
        # Y_[:,group[param1_test]],y_gen,nh))
        x_gen_, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]],
        Y_[:,group[param1_test]],y_gen,nh)
        n = length(x_gen_)
        # println(x_gen_)
        x_gen_j = []
        push!(x_gen_j,x_gen_[[1:Int(n/3)...]][sec])
        push!(x_gen_j,x_gen_[[Int(n/3)+1:Int(n/3)*2...]][sec])
        push!(x_gen_j,x_gen_[[(Int(n/3)*2)+1:n...]][sec])
        push!(x_gen,x_gen_j)
    end
    for i in 2:n_
        for j in 1:3
            Δ = x_gen[i][j][begin]-x_gen[i-1][j][end]
            x_gen[i][j] = x_gen[i][j].-Δ
        end
        
    end



    x_gen_ = []
    for i in 1:3, j in 1:n_
        if j==n_
            append!(x_gen_,x_gen[j][i])
        else
            pop!(x_gen[j][i])
            append!(x_gen_,x_gen[j][i])
        end
    end
    return x_gen_
end

function NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    Κ_test(X) = Κ(x_test,X)
    g_i = Κ_test.(eachcol(X))
    n = length(g_i)
    ḡ_i = g_i - (1/n)*G*ones(n,1) - (1/n)*ones(n,n)*g_i +
    ((1/n^2)*ones(1,n)*G*ones(n,1))[1]*ones(n,1)
    z_i = real.(U_'*ḡ_i)
    # scatter!(eachrow(z_i)..., hover = [i for i in 1:lastindex(eachcol(Z_))])
    Z_i = hcat(Z_,z_i)
    Last = lastindex(eachcol(Z_i))
    d_G_i, prev_ = Dijkstra(Z_i,Last,neighbors)
    D_G_i = vcat(hcat(D_G_sym,d_G_i[[1:n...]]),d_G_i')
    D_G_sq_i = D_G_i.^2
    C_i = I - (1/(n+1))*ones(n+1,n+1)
    B_i = -0.5*C_i*D_G_sq_i*C_i
    f(λ) = -real(λ)
    _iso_i = eigen(B_i, sortby=f)
    Λ_iso_i = real.(_iso_i.values)
    U_iso_i = real.(_iso_i.vectors)

    Λ_iso_m_i = diagm(sqrt.(Λ_iso_i[[1,2]]))
    U_iso_m_i = U_iso_i[:,[1,2]]

    X_iso_i = U_iso_m_i*Λ_iso_m_i
    X_iso_i = X_iso_i'
    return X_iso_i[:,Last]
end