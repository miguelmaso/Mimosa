using CSV
using Tables
using DataFrames
using Statistics
using LinearAlgebra
using Plots
using Optim
using NearestNeighbors
plotlyjs()

function DistG_WPlot(i,j,k,k0,data,kdtree)
    t1 = time()
    if i == j
        return D_G = 0.0
    end
    point1 = data[:,i]
    point2 = data[:,j]
    idxs_list = [i]
    Dₑ = norm(point2-point1)
    println("Euclidean distance $Dₑ")
    display(scatter(eachrow(point1)...))
    display(scatter!(eachrow(point2)...))
    display(plot!(eachrow(data[:,[i,j]])...))
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
        D_pair = norm(data[:,idx]-point0)
        D_G += D_pair
        println("Accumulative distance $D_G")
        println("Pair distance $D_pair")
        point0 = data[:,idx]
    end
    println(time()-t1)
    return D_G, Dₑ
end

function DistG(i,j,k,k0,data,kdtree)
    t1 = time()
    if i == j
        return D_G = 0.0
    end
    pair = (i,j)
    println("Current $pair")
    point1 = data[:,i]
    point2 = data[:,j]
    idxs_list = [i]
    Dₑ = norm(point2-point1)
    # println("Euclidean distance $Dₑ")
    # display(scatter(eachrow(point1)...))
    # display(scatter!(eachrow(point2)...))
    # display(plot!(eachrow(data[:,[1,2]])...))
    idxs, dists = knn(kdtree, point1, k, false, x -> x==1)
    # display(scatter!(eachrow(data[:,idxs])...))
    indices = sortperm([norm(point2-data[:,id]) for id in idxs])
    idxs = idxs[indices[[1:k0...]]]
    push!(idxs_list,idxs[1])
    # display(scatter!(eachrow(data[:,idxs])...))
    Dist = norm(point2-data[:,idxs[1]])
    while Dist>0.0
        idxs_ = []
        for idx in idxs
            append!(idxs_,idx)
        end
        idxs = idxs_
        indices = sortperm([norm(point2-data[:,id]) for id in idxs])
        idxs = idxs[indices[[1:k0...]]]
        # display(scatter!(eachrow(data[:,idxs])...))
        Dist = norm(point2-data[:,idxs[1]])
        push!(idxs_list,idxs[1])
        # println("How close: $Dist")
        idxs, dists = knn(kdtree, data[:,idxs], k, true, x -> maximum(isequal.(x,idxs)))
        
        # display(scatter!(eachrow(data[:,idxs_])...))
        # println(i)
    end
    # display(plot!(eachrow(data[:,idxs_list])...))
    point0 = data[:,idxs_list[1]]
    D_G = 0
    for idx in idxs_list
        D_G += norm(data[:,idx]-point0)
        point0 = data[:,idx]
    end
    println(time()-t1)
    return D_G
end

function isomap0(Distance,X,kdtree,k,k_n)
    m, n = size(X)
    G = [Distance(i,j,k_n,1,X,kdtree) for i in 1:n, j in 1:n]
    G = 0.5*(G+G')
    H = I - (1/n)*ones(n,n)
    G = 0.5*H'*G*H
    f(λ)=-abs(λ)
    Λ_ = eigen(G,sortby=f)
    Λ = Λ_.values
    U = Λ_.vectors
    # U_ = U[:,1:k]
    return Λ, U, G
end

function isomap1(neighbors,Z_)
    n = length(eachcol(Z_))
    D_G = [[] for i in 1:n]
    count = []
    count2 = []
    list = [1:n...]
    Threads.@threads for i in 1:n
        d_G, prev_ = Dijkstra(Z_,i,neighbors)
        D_G[i] = d_G
        deleteat!(list, findall(x->x==i,list))
        print("\r$((100-round(length(list)/n,digits=2)*100)) %             ")
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

function ReadData(n_parts,Phi,d)
    X = []
    for i in 0:n_parts, j in Phi
        push!(X,CSV.File("data/csv/EM_PB_10S_Phi$j/EM_PB_10S_$i.csv") |> Tables.matrix)
    end
    X = hcat(X...)
    if d
        m, n = size(X)
        z = zeros(1,n)
        X1 = vcat(X,z)
        X2 = vcat(z,X)
        X3 = (1/0.001)*(X1-X2)
        X = X3[[2:m...],:]
    end
    return X
end

function ReadData_RTS_STS(RTS_parts,STS_parts,Phi)
    X = []
    for i in 0:RTS_parts, j in Phi
        push!(X,CSV.File("data/csv/EM_PB_10S_Phi$(j)_Random/EM_PB_10S_$i.csv") |> Tables.matrix)
    end
    conf_ = []
    for j in Phi
        push!(conf_,CSV.File("data/csv/EM_PB_10S_Phi$(j)_Random/Config_N1422_EM_PB_10S.csv") |> Tables.matrix)
    end
    conf_ = hcat(conf_...)
    X_ = hcat(X...)
    n_ = lastindex(eachcol(X_))
    conf = conf_[:,[1:n_...]]
    if STS_parts>0
        for i in 1:STS_parts, j in Phi
            push!(X,CSV.File("data/csv/EM_PB_10S_Phi$(j)_SmartTS/EM_PB_10S__$i.csv") |> Tables.matrix)
        end
        conf_ = []
        for j in Phi
            push!(conf_,CSV.File("data/csv/EM_PB_10S_Phi$(j)_SmartTS/Config_EM_PB_10S_Smart800.csv") |> Tables.matrix)
        end
        conf_ = hcat(conf_...)
        X = hcat(X...)
        conf = hcat(conf,conf_[:,[1:lastindex(eachcol(X))-n_...]])
    else
        X = X_
    end

    return X, conf
end

function ReadData_RTS_STS(RTS_parts,STS_parts,RTS_colums,STS_columns,Phi)
    X = []
    for i in 0:RTS_parts, j in Phi
        push!(X,CSV.File("data/csv/EM_PB_10S_Phi$(j)_Random/EM_PB_10S_$i.csv") |> Tables.matrix)
    end
    conf_ = []
    for j in Phi
        push!(conf_,CSV.File("data/csv/EM_PB_10S_Phi$(j)_Random/Config_N1422_EM_PB_10S.csv") |> Tables.matrix)
    end
    conf_ = hcat(conf_...)
    X_ = hcat(X...)
    X_ = X_[:,[1:RTS_colums...]]
    conf = conf_[:,[1:RTS_colums...]]
    X__ = []
    if STS_parts>0
        for i in 1:STS_parts, j in Phi
            push!(X__,CSV.File("data/csv/EM_PB_10S_Phi$(j)_SmartTS/EM_PB_10S__$i.csv") |> Tables.matrix)
        end
        X___ = hcat(X__...)
        X___ = X___[:,[1:STS_columns...]]
        conf_ = []
        for j in Phi
            push!(conf_,CSV.File("data/csv/EM_PB_10S_Phi$(j)_SmartTS/Config_EM_PB_10S_Smart800.csv") |> Tables.matrix)
        end
        conf_ = hcat(conf_...)
        conf = hcat(conf,conf_[:,[1:STS_columns...]])
        X = hcat(X_,X___)
    else
        X = X_
    end

    return X, conf
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

function PlotSetup(X_,k)
    X_pl = []
    for i in 1:k
        push!(X_pl,X_[i,:])
    end
    return X_pl
end

function PlotSetupN(X_,k)
    X_pl = []
    for i in 1:k
        push!(X_pl,X_[i,:])
    end
    N_pl = [i for i in 1:length(X_pl[1])]
    pushfirst!(X_pl,N_pl)
    return X_pl
end

function plotkPOD(Ḡ,U_,k,d)
    m, n = size(Ḡ)
    Z_ = U_'*Ḡ #Corregir uso de G y usar la ecuación 2.28 del capitulo del libro
    # Z_ = abs.(Z_)
    Z_ = real.(Z_)
    Z_pl = PlotSetup(Z_,k)
    s = scatter(Z_pl...,xlabel="z1",ylabel="z2",zlabel="z3", hover= [i for i in 1:n])#, markercolor= [Int(ceil(i/l)) for i in 1:n])
    display(s)
end

function execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Λ_t = sum(Λ)
    Λ = (1/Λ_t)*Λ
    Λ_s = round.(100*Λ[[i for i in 1:10]])
    p = plot(real.(Λ_s),type="bar",xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
    display(p)
    if k<=3
        plotkPOD(Ḡ,U_,k,d)
    else
        plotkPOD(Ḡ,U_,3,d)
    end
    return X, U_, Ḡ
end

function ReverseMap(X,Z_,z,d,n)
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
    if d
        y = [0.004]
        n = length(x)
        for i in 1:n
            push!(y,y[i]+0.001*real.(x[i]))
        end
        x=y
    end
    return x, w_ns, Z_ns
end

function ReverseMap2(y_gen,conf_gen,Y_,X,conf,nh,t)
    x_gen = []
    for j in 1:10
        param1 = []
        for i in 1:n
            push!(param1,conf[j,i])
        end
        sort_param1 = sortperm(param1)
        group_param = 0
        group = [[],[],[]]
        count = 1
        for i in 1:n
            if param1[sort_param1[i]]==group_param
                push!(group[count],sort_param1[i])
            else
                count += 1
                group_param += 1
                push!(group[count],sort_param1[i])
            end
        end
        param1_test = conf_gen[j]
        param1_test += 1
        sec = [((10*(j-1)+1)):((10*j)+1)...]
        # if j==10
        #     append!(sec,101)
        # end
        x_gen_j, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]][sec,:],
        Y_[:,group[param1_test]],y_gen,false,nh)
        push!(x_gen,x_gen_j)
    end
    for i in 2:10
        Δ = x_gen[i][begin]-x_gen[i-1][end]
        x_gen[i] = x_gen[i].-Δ
        θ1 = atan((x_gen[i-1][end] - x_gen[i-1][end-1])/0.001)
        θ2 = atan((x_gen[i][begin+1] - x_gen[i][begin])/0.001)
        Δθ = θ1 - θ2
        # Δθ = Δθ*1.5
        # display((Δθ*(180/pi)))
        rot = [cos(Δθ) -sin(Δθ); sin(Δθ) cos(Δθ)]
        # display(rot)
        x_gen_i = x_gen[i]
        b = x_gen_i[1]
        x_gen_i = x_gen_i.- b
        x_ = [0:10...]
        x_ = x_./1000
        X_ = hcat(x_,x_gen_i)
        X_ = X_'
        # # Poly2(z) = [1 z z^2 z^3]
        # Poly2(z) = [1 z z^2]
        # A = Poly2.(X_[1,:])
        # A = reduce(vcat,A)
        # c = pinv(A)*X_[2,:]
        # # Poly_d(z) = [0 1 2*z 3*z^2]
        # Poly_d(z) = [0 1 2*z]
        # A = Poly_d(-t)
        # θ2 = A*c
        # Δθ = θ1 - θ2[1]
        # rot = [cos(Δθ) -sin(Δθ); sin(Δθ) cos(Δθ)]
        # plot(eachrow(X_)...)
        X_ = rot*X_
        # plot!(eachrow(X_)...)
        Poly(z) = [1 z z^2 z^3]
        A = Poly.(X_[1,:])
        A = reduce(vcat,A)
        c = pinv(A)*X_[2,:]
        A = Poly.(x_)
        A = reduce(vcat,A)
        x_gen[i] = A*c
        # display(plot!(x_,x_gen[i],title="$i"))
        x_gen[i] .+= b
    end
    for i in 1:lastindex(x_gen)-1
        pop!(x_gen[i])
    end
    x_gen_0 = reduce(append!,x_gen)
    x_gen_0 =  1.10145.*x_gen_0
    return x_gen_0
end

function RotM(v1,v2)
    v = cross(v1,v2)
    vₓ = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
    c = dot(v1,v2)
    R = I + vₓ + (1/(1+c))*vₓ*vₓ
    return R
end

function VectorSearch(Z_,conf)
    z = Z_[:,[1:22...]]
    v = [z[:,i]-z[:,1] for i in 2:21]
    z = z[:,1]
    for i in 1:lastindex(conf)
        # v_ = v[i]
        # v_[1] = conf[i]*v_[1]
        if conf[i] == 0
            z = z
        elseif conf[i]==1
            z = z + v[lastindex(conf)-i+1]
        elseif conf[i]==2
            z = z + v[2*lastindex(conf)-i+1]
        end
    end
    return z
end

function VectorSearch(Z_,conf,∇zᵤ)
    z = Z_[:,[end-21:end...]]
    v = [z[:,i]-z[:,1] for i in 2:21]
    z = z[:,1]
    R = I
    u_0 = ∇zᵤ(z[1],z[2])
    for i in 1:lastindex(conf)
        # v_ = v[i]
        # v_[1] = conf[i]*v_[1]
        if conf[i] == 0
            z = z
        elseif conf[i]==1
            z = z + R*v[lastindex(conf)-i+1]
        elseif conf[i]==2
            z = z + R*v[2*lastindex(conf)-i+1]
        end
        # println(R)
        u_1 = ∇zᵤ(z[1],z[2])
        R = RotM(u_1,u_0)
        # u_0 = u_1
    end
    return z
end

function Objective1(β)
    println("Kernel parameter: $β")
    P = [2000] #,4000]
    n_parts = 8
    k = 3
    d = false
    # Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Κ(X1,X2) = (Der2nd(X1,0.001)'*Der2nd(X2,0.001) + β)^2
    # Κ(X1,X2) = (X1'*X2 + β)^2
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)
    final = lastindex(eachcol(Z_))
    D = Dif(Z_,final-21,final-12,final-11,final)
    return D
    # return 1-(abs(maximum(Z_[2,:]))/abs(maximum(Z_[1,:])))
end

function Objective2(β)
    println("Kernel parameter: $β")
    P = [2000] #,4000]
    n_parts = 6
    k = 3
    d = false
    # Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Κ(X1,X2) = (X1'*X2 + β)^2
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    R = []
    err = []
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)

    # -------Curve fitting------
    # Quadric surface explicit in z3 of the form z3 = A + B*z1^2 + Cz2^2 + D*z1 + E*z2
    Quadric(z) = [1 z[1]^2 z[2]^2 z[1] z[2]] 

    # least squares fit of the curve to the TS in the RS
    A = [Quadric(Z) for Z in eachcol(Z_)]
    A = reduce(vcat,A)
    c = pinv(A)*Z_[3,:]
    
    #Definition of surface and unit gradient
    z3(z1,z2) = c[1] + c[2]*z1^2 + c[3]*z2^2 + c[4]*z1 + c[5]*z2
    ∇zᵤ(z1,z2) = (1/norm([2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]))*[2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]

    SS_reg = sqrt(dot(Z_[3,:]-z3.(Z_[1,:],Z_[2,:]),Z_[3,:]-z3.(Z_[1,:],Z_[2,:])))
    avr = sum(Z_[3,:])/length(Z_[3,:])
    SS_tot = sqrt(dot(Z_[3,:].-avr,Z_[3,:].-avr))
    R_sq = 1-SS_reg/SS_tot
    append!(R,sqrt(R_sq))
    println("R1 = $R")

    # Whole RS found using VectorSearch and considering ∇zᵤ
    z_ = []
    for c in eachcol(conf)
        z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
        push!(z_,z)
    end
    z_ = reduce(hcat,z_)

    # evaluation of the distributted error - Applicable if using the whole RS as the TS
    append!(err,norm(Z_-z_)/norm(Z_))
    println("Err1 = $err")

    Z_c = Z_[3,:]
    Z_[3,:] = Z_[2,:]
    Z_[2,:] = Z_c
    
    A = [Quadric(Z) for Z in eachcol(Z_)]
    A = reduce(vcat,A)
    c = pinv(A)*Z_[3,:]
    
    #Definition of surface and unit gradient
    z3(z1,z2) = c[1] + c[2]*z1^2 + c[3]*z2^2 + c[4]*z1 + c[5]*z2
    ∇zᵤ(z1,z2) = (1/norm([2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]))*[2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]

    SS_reg = sqrt(dot(Z_[3,:]-z3.(Z_[1,:],Z_[2,:]),Z_[3,:]-z3.(Z_[1,:],Z_[2,:])))
    avr = sum(Z_[3,:])/length(Z_[3,:])
    SS_tot = sqrt(dot(Z_[3,:].-avr,Z_[3,:].-avr))
    R_sq = 1-SS_reg/SS_tot
    append!(R,sqrt(R_sq))
    println("R2 = $R")

    # Whole RS found using VectorSearch and considering ∇zᵤ
    z_ = []
    for c in eachcol(conf)
        z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
        push!(z_,z)
    end
    z_ = reduce(hcat,z_)

    # evaluation of the distributted error - Applicable if using the whole RS as the TS
    append!(err,norm(Z_-z_)/norm(Z_))
    println("Err2 = $err")

    if R[1]>R[2]
        err = err[1]
        println("Err1 & R1")
    else
        err = err[2]
        println("Err2 & R2")
    end
    
    return err
end

function Objective3(β)
    println("Kernel parameter: $β")
    P = [2000] #,4000]
    n_parts = 6
    k = 3
    d = false
    # Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Κ(X1,X2) = (X1'*X2 + β)^2
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    R = []
    err = []
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)

    kdtree = KDTree(Z_; leafsize = 10)
    data = Z_
    k0 = 1
    k_n = 30
    k1 = 2
    Λ, U, G = isomap0(DistG,data,kdtree,k1,k_n)
    Σ = diagm(real.(sqrt.(complex(-Λ))))
    Σ_ = Σ[:,[1:2...]]
    Z_ = real.(Σ_'*U)
    
    # Whole RS found using VectorSearch
    z_ = []
    for c in eachcol(conf)
        z = VectorSearch(Z_,c); z = real.(z)
        push!(z_,z)
    end
    z_ = reduce(hcat,z_)

    # evaluation of the distributted error - Applicable if using the whole RS as the TS
    err = norm(Z_-z_)/norm(Z_)
    
    return err
end

function Objective4(β)
    println("Kernel parameter: $β")
    P = [4000] #,4000]
    n_parts = 8
    k = 3
    d = false
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    # Κ(X1,X2) = (X1'*X2 + β)^2
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)
    D = -Dist(Z_,601,600)
    return D
    # return 1-(abs(maximum(Z_[2,:]))/abs(maximum(Z_[1,:])))
end

function Objective5(β)
    println("Kernel parameter: $β")
    P = [2000] #,4000]
    n_parts = 8
    k = 3
    d = false
    # Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    # Κ(X1,X2) = (X1'*X2 + β)^2
    # Κ(X1,X2) = (Der2nd(X1,0.001)'*Der2nd(X2,0.001) + β)^2
    Κ(X1,X2) = exp(-β*(dot(Der2nd(X1,0.001)-Der2nd(X2,0.001),Der2nd(X1,0.001)-Der2nd(X2,0.001))))
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)
    Last = lastindex(eachcol(Z_))
    d_G, prev_ = Dijkstra(Z_,1,25) # conf 0000000000
    d1 = d_G[10] # conf 0100000000
    d_G, prev_ = Dijkstra(Z_,11,25) # conf 1000000000
    d2 = d_G[22] # conf 1100000000
    D = abs(d2-d1)/d1
    return D
    # return 1-(abs(maximum(Z_[2,:]))/abs(maximum(Z_[1,:])))
end

function Objective7(β)
    println("Kernel parameter: $β")
    P = [2000] #,4000]
    n_parts = 8
    k = 3
    d = false
    # Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Κ(X1,X2) = (X1'*X2 + β)^2
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)
    neighbors = 25
    to = time()
    Y_, D_G_sym = isomap1(neighbors,Z_)
    println("elapsed time for isomap")
    println(time()-to)
    Y_gen = []
    Last = lastindex(eachcol(Y_))
    for i in 1:Last
        y_ = VectorSearch(Y_,conf[:,i])
        push!(Y_gen,y_)
    end
    Y_gen = reduce(hcat,Y_gen)
    D = norm(Y_gen-Y_)/norm(Y_)
    println("Err = $D")
    return D
end

function Objective8(β)
    println("Kernel parameter: $β")
    P = [2000] #,4000]
    n_parts = 8
    k = 3
    d = false
    # Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Κ(X1,X2) = (Der2nd(X1,0.001)'*Der2nd(X2,0.001) + β)^2
    # Κ(X1,X2) = (X1'*X2 + β)^2
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Λ = real.(Λ)
    Λ_t = sum(Λ)
    Λ_3 = sum(Λ[[1,2,3]])
    Res = Λ_3/Λ_t
    println("Info in first 3 directcions: $Res")
    return 1-Res
    # return 1-(abs(maximum(Z_[2,:]))/abs(maximum(Z_[1,:])))
end


function NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    Κ_test(X) = Κ(x_test,X)
    g_i = Κ_test.(eachcol(X))
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

Dist(Z_,i1,i2) = sqrt(dot(real.(Z_[:,i1]-Z_[:,i2]),real.(Z_[:,i1]-Z_[:,i2])))
Dif(Z_,i,j,k,l) = abs((Dist(Z_,k,l)-Dist(Z_,i,j))/Dist(Z_,i,j))
