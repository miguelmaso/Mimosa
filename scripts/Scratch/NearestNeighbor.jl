using NearestNeighbors
using LinearAlgebra
using Plots

function DistG(i,j,k,k0,data,kdtree)
    point1 = data[:,i]
    point2 = data[:,j]
    idxs_list = [i]
    Dₑ = norm(point2-point1)
    println("Euclidean distance $Dₑ")
    display(scatter(eachrow(point1)...))
    display(scatter!(eachrow(point2)...))
    display(plot!(eachrow(data[:,[1,2]])...))
    idxs, dists = knn(kdtree, point1, k, false, x -> x==1)
    # display(scatter!(eachrow(data[:,idxs])...))
    indices = sortperm([norm(point2-data[:,id]) for id in idxs])
    idxs = idxs[indices[[1:k0...]]]
    push!(idxs_list,idxs[1])
    display(scatter!(eachrow(data[:,idxs])...))
    Dist = norm(point2-data[:,idxs[1]])
    while Dist>0.0
        idxs_ = []
        for idx in idxs
            append!(idxs_,idx)
        end
        idxs = idxs_
        indices = sortperm([norm(point2-data[:,id]) for id in idxs])
        idxs = idxs[indices[[1:k0...]]]
        display(scatter!(eachrow(data[:,idxs])...))
        Dist = norm(point2-data[:,idxs[1]])
        push!(idxs_list,idxs[1])
        println("How close: $Dist")
        idxs, dists = knn(kdtree, data[:,idxs], k, true, x -> maximum(isequal.(x,idxs)))
        
        # display(scatter!(eachrow(data[:,idxs_])...))
        # println(i)
    end
    display(plot!(eachrow(data[:,idxs_list])...))
    point0 = data[:,idxs_list[1]]
    D_G = 0
    for idx in idxs_list
        D_G += norm(data[:,idx]-point0)
        point0 = data[:,idx]
    end
    return D_G, Dₑ
end

function Dijkstra(data,i)
    m, n = size(data)
    dist = [Inf64 for i in 1:n]
    prev = [0 for i in 1:n]
    nodes = data
    kdtree = KDTree(nodes; leafsize = 10)
    dist[i] = 0
    visited = []
    dist_ = copy(dist)
    nieghbors = 5
    while length(visited)<n
        println(length(visited))
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


data_size = 10^3
i = 100

dist, prev = Dijkstra(data,i)

D_G = []
for i in 1:data_size
    dist, prev = Dijkstra(data,i)
    push!(D_G,dist)
end

plotlyjs()
j = 2
scatter(eachrow(nodes)...)
scatter!(eachrow(nodes[:,[i]])...)
scatter!(eachrow(nodes[:,[j]])...)
prev_ = j
while prev_ ≠ i
    display(scatter!(eachrow(nodes[:,[prev[prev_]]])...))
    prev_ = prev[prev_]
end


# D_G, Dₑ = DistG(i,j,k,k0,data,kdtree)
# err = []
# for l in 1:100
#     k = Int(round((l/1000)*data_size))
#     D_G, Dₑ = DistG(i,j,k,k0,data,kdtree)
#     push!(err,D_G/Dₑ-1)
# end
# plot(err)