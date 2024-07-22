using CSV
using Tables
using LinearAlgebra
using Plots
using NearestNeighbors

function ReadData2(pot,parts)
    j = pot
    i = 1
    _X = CSV.File("data/csv/EM_TB_ST4SL2_Phi$j/EM_TB_St4_Sl2_$i.csv") |> Tables.matrix
    _X = _X'
    n = 64
    X_ = []
    push!(X_,_X[:,[1:n...]])
    push!(X_,_X[:,[n+1:2*n...]])
    push!(X_,_X[:,[2*n+1:3*n...]])
    # X_[1] = hcat(X_[1],_X[:,[1:n...]])
    # X_[2] = hcat(X_[2],_X[:,[n+1:2*n...]])
    # X_[3] = hcat(X_[3],_X[:,[2*n+1:3*n...]])
    for i in 2:parts
        _X = CSV.File("data/csv/EM_TB_ST4SL2_Phi$j/EM_TB_St4_Sl2_$i.csv") |> Tables.matrix
        _X = _X'
        n = 64
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