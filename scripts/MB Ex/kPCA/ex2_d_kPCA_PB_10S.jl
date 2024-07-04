n_p, P, d = 6, [2000], false
# Training Set in the full-order space
X = ReadData(n_p, P, d)
# Number of principal directions to be considered
k = 3
#Generation of full set of parameters in order
conf = CSV.File("data/csv/EM_PB_10S_Phi2000/Config_N622_EM_PB_10S.csv") |> Tables.matrix

#Gaussian Kernel
# Κ(X1,X2) = exp(-3.0*(dot(X1-X2,X1-X2)))
# Polynomial Kernel
Κ(X1,X2) = (X1'*X2 + 0.31)^2
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)

# Training Set in the reduced space
Z_ = real.(U_'*Ḡ)
scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3",
color = [c[1] for c in eachcol(conf)])
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))],
xlims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
ylims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
zlims=(minimum(Z_[1,:]),maximum(Z_[1,:])))
#Generation of full set of parameters in order
conf = CSV.File("data/csv/EM_PB_10S_Phi2000/Config_N622_EM_PB_10S.csv") |> Tables.matrix

β_min = optimize(Objective5, 0.00, 25, GoldenSection())
β = β_min.minimizer
Κ(X1,X2) = (X1'*X2 + β)^2
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3",
color = [c[1] for c in eachcol(conf)])



d_G, prev_ = Dijkstra(Z_,600,5)

d_G[274]



neighbors = 25
n = length(eachcol(Z_))
D_G = [[] for i in 1:n]
to = time()
Threads.@threads for i in 1:n
    d_G, prev_ = Dijkstra(Z_,i,neighbors)
    D_G[i] = d_G
    # print("\r------------------------ $i")
end
t = time()-to
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
scatter(eachcol(X_iso)..., hover = [i for i in 1:n],
xlabel="x_iso_1",ylabel="x_iso2")# , color = [c[1] for c in eachcol(conf)])

x_iso = []
for i in 1:n
    x_ = VectorSearch(X_iso',conf[:,i])
    push!(x_iso,x_)
end
x_iso = reduce(hcat,x_iso)
scatter!(eachcol(x_iso')..., hover = [i for i in 1:n],
xlabel="x_iso_1",ylabel="x_iso2")
norm(X_iso-x_iso')/norm(X_iso)

d_err = [norm(X_iso[i,:]-x_iso[:,i]) for i in 1:lastindex(eachrow(X_iso))]
r_X_iso = [norm(X) for X in eachrow(X_iso)]
scatter(r_X_iso,d_err, hover = [i for i in 1:n], legend=false)
scatter!([x_[1]],[x_[2]], hover = [i for i in 1:n],
xlabel="x_iso_1",ylabel="x_iso2")

function Objective6_(x,x_iso,X_iso)
    θ = x[1]
    α = x[2]
    rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
    scale = [α 0;0 α]
    x_iso = rot*x_iso
    x_iso = scale*x_iso
    return norm(X_iso-x_iso')/norm(X_iso)
end

Objective6(x) = Objective6_(x,x_iso,X_iso)
x0 = [0.0,1.0]
_min = optimize(Objective6, x0, NelderMead())
θ = _min.minimizer[1]
α = _min.minimizer[2]
rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
scale = [α 0;0 α]
x_iso = rot*x_iso
x_iso = scale*x_iso

X_test = []
for i in 1:3
    push!(X_test,CSV.File("data/csv/EM_PB_10S_Phi2000_test/EM_PB_10S_$i.csv") |> Tables.matrix)
end
X_test = hcat(X_test...)

conf_test = CSV.File("data/csv/EM_PB_10S_Phi2000_test/Config_N60_EM_PB_10S.csv") |> Tables.matrix


x_ = VectorSearch(X_iso',conf_test[:,10])
x_ = rot*x_
x_ = scale*x_
scatter!([X_iso[102,1]],[X_iso[102,2]])
ReverseMap(X,Z_,z,d,n)
x, w_ns, Z_ns = ReverseMap(X,X_iso',x_,false,8)

plot(x,label="Generated X_test[10]", legend=:bottomright)
plot!(X_test[:,10],label="FOD X_test[10]")

norm(X_test[:,10]-x)/norm(X_test[:,10])

conf_complete = []
for x1 in 0:2, x2 in 0:2, x3 in 0:2, x4 in 0:2, x5 in 0:2, x6 in 0:2, x7 in 0:2, x8 in 0:2, x9 in 0:2, x10 in 0:2
  push!(conf_complete,[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
end
conf_complete = reduce(hcat,conf_complete)

x_complete = []
for c in eachcol(conf_complete)
    x = VectorSearch(X_iso',c); x = real.(x)
    push!(x_complete,x)
end
x_complete = reduce(hcat,x_complete)
x_complete = rot*x_complete
x_complete = scale*x_complete
x__ = [[] for i in 1:lastindex(eachcol(x_complete))]
Threads.@threads for i in 1:lastindex(eachcol(x_complete))
    x_ = x_complete[:,i]
    x, w_ns, Z_ns = ReverseMap(X,X_iso',x_,false,8)
    Κ_test(X) = Κ(x_test,X)
    g_i = Κ_test.(eachcol(X))
    ḡ_i = g_i - (1/n)*G*ones(n,1) - (1/n)*ones(n,n)*g_i +
    ((1/n^2)*ones(1,n)*G*ones(n,1))[1]*ones(n,1)
    z_i = real.(U_'*ḡ_i)
    # scatter!(eachrow(z_i)..., hover = [i for i in 1:lastindex(eachcol(Z_))])
    Z_i = hcat(Z_,z_i)
    d_G_i, prev_ = Dijkstra(Z_i,623,neighbors)
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
    x__[i] = X_iso_i[:,623]
end
conf_test = CSV.File("data/csv/EM_PB_10S_Phi2000_test/Config_N60_EM_PB_10S.csv") |> Tables.matrix

kdtree = KDTree(x_complete; leafsize = 10)
idxs, dists = knn(kdtree, X_iso_i[:,623], 50, true)

scatter!(eachrow(x_complete[:,[idxs...]])..., hover = [i for i in 1:n],
xlabel="x_iso_1",ylabel="x_iso2")


id_conf_list = [0 for i in 1:30]
n_test = 1
for n_test in 1:30
    x_test = X_test[:,n_test]
    Κ_test(X) = Κ(x_test,X)
    g_i = Κ_test.(eachcol(X))
    ḡ_i = g_i - (1/n)*G*ones(n,1) - (1/n)*ones(n,n)*g_i +
    ((1/n^2)*ones(1,n)*G*ones(n,1))[1]*ones(n,1)
    z_i = real.(U_'*ḡ_i)
    # scatter!(eachrow(z_i)..., hover = [i for i in 1:lastindex(eachcol(Z_))])
    Z_i = hcat(Z_,z_i)
    d_G_i, prev_ = Dijkstra(Z_i,623,neighbors)
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
    scatter!([X_iso_i[1,623]],[X_iso_i[2,623]])


    kdtree = KDTree(x_complete; leafsize = 10)
    idxs, dists = knn(kdtree, X_iso_i[:,n+1], 200, true)
    # scatter!([x_complete[1,idxs]],[x_complete[2,idxs]])
    x__ = copy(x_complete)
    Threads.@threads for i in 1:200
        x_ = x_complete[:,idxs[i]]
        x, w_ns, Z_ns = ReverseMap(X,X_iso',x_,false,8)
        Κ_test(X) = Κ(x,X)
        g_i = Κ_test.(eachcol(X))
        ḡ_i = g_i - (1/n)*G*ones(n,1) - (1/n)*ones(n,n)*g_i +
        ((1/n^2)*ones(1,n)*G*ones(n,1))[1]*ones(n,1)
        z_i = real.(U_'*ḡ_i)
        # scatter!(eachrow(z_i)..., hover = [i for i in 1:lastindex(eachcol(Z_))])
        Z_i = hcat(Z_,z_i)
        d_G_i, prev_ = Dijkstra(Z_i,623,neighbors)
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
        x__[:,idxs[i]] = X_iso_i[:,623]
    end
    scatter!([x__[1,idxs]],[x__[2,idxs]])
    kdtree = KDTree(x__; leafsize = 10)
    idxs, dists = knn(kdtree, X_iso_i[:,n+1], 200, true)
    scatter!([x__[1,idxs]],[x__[2,idxs]])

    id_conf = 0
    for i in 1:lastindex(idxs)
        if conf_complete[:,idxs[i]]==conf_test[:,n_test]
            id_conf = i
            println(i)
        end
    end
    conf_complete[:,idxs[id_conf]]
    conf_test[:,n_test]
    id_conf_list[n_test] = id_conf
end

plot(id_conf_list, xlabel="Test configuration ID", 
ylabel="neighbor number of the test configuration", type=:bar)
savefig("data/Figs/TestConfigVSNeighborNumber.png")
mean(id_conf_list[[1:25...]])
conf_test[:,27]



n_test = 10
x_test = X_test[:,n_test]
Κ_test(X) = Κ(x_test,X)
g_i = Κ_test.(eachcol(X))
ḡ_i = g_i - (1/n)*G*ones(n,1) - (1/n)*ones(n,n)*g_i +
((1/n^2)*ones(1,n)*G*ones(n,1))[1]*ones(n,1)
z_i = real.(U_'*ḡ_i)
scatter!(eachrow(z_i)..., hover = [i for i in 1:lastindex(eachcol(Z_))])
Z_i = hcat(Z_,z_i)
d_G_i, prev_ = Dijkstra(Z_i,623,neighbors)
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
X_iso_i[:,623]
scatter!([X_iso_i[1,623]],[X_iso_i[2,623]], hover = [i for i in 1:n],
xlabel="x_iso_1",ylabel="x_iso2")

x_iso_complete = []
    for i in 1:n
        x_ = VectorSearch(X_iso',conf[:,i])
        push!(x_iso_complete,x_)
    end
x_iso_complete = reduce(hcat,x_iso_complete)
# ---------------- Neighbors test (accuracy vs time)--------------
Neighbors = [5,10,15,20,25,30,35]
err = []
ti = []
for neighbors in Neighbors
    println(neighbors)
    n = length(eachcol(Z_))
    D_G = [[] for i in 1:n]
    to = time()
    Threads.@threads for i in 1:n
        d_G, prev_ = Dijkstra(Z_,i,neighbors)
        D_G[i] = d_G
        # print("\r------------------------ $i")
    end
    t = time()-to
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
    # scatter(eachcol(X_iso)..., hover = [i for i in 1:n])

    x_iso = []
    for i in 1:n
        x_ = VectorSearch(X_iso',conf[:,i])
        push!(x_iso,x_)
    end
    x_iso = reduce(hcat,x_iso)
    # scatter!(eachrow(x_iso)..., hover = [i for i in 1:n])
    push!(err,norm(x_iso - X_iso')/norm(X_iso))
    push!(ti,t)
end
gr()
a = plot(Neighbors,err, color = :red, xlabel="neighbors",
ylabel="err - norm(x_iso_gen - X_iso')/norm(X_iso)", legend = :false, size = (800,500),
marker = :ltriangle, markersize = 8)
b = plot!(twinx(),Neighbors,ti, color = :blue, ylabel="time",
legend=:false, marker = :rtriangle, markersize = 8)
savefig("data/Figs/isomap_err_TS.png")