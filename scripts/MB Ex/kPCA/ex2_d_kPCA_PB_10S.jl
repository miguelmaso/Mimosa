n_p, P, d = 6, [2000], false
# Training Set in the full-order space
X = ReadData(n_p, P, d)
# Number of principal directions to be considered
k = 3
#Generation of full set of parameters in order
conf = CSV.File("data/csv/EM_PB_10S_Phi2000/Config_N622_EM_PB_10S.csv") |> Tables.matrix

β_min = optimize(Objective7, 0.00, 1, GoldenSection())
β = β_min.minimizer
β = 0.1538588229886658 # Obtained by optimizing the error of the whole procedure to regenerate the TS via Vector Search
Κ(X1,X2) = (X1'*X2 + β)^2
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3")
scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3",
color = [c[1] for c in eachcol(conf)])
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))],
xlims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
ylims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
zlims=(minimum(Z_[1,:]),maximum(Z_[1,:])))

neighbors = 25
Y_, D_G_sym = isomap1(neighbors,Z_)
scatter(eachrow(Y_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="y1",ylabel="y2")
n = lastindex(eachcol(Y_))
Y_gen = []
for i in 1:n
    y_ = VectorSearch(Y_,conf[:,i])
    push!(Y_gen,y_)
end
Y_gen = reduce(hcat,Y_gen)

scatter!(eachrow(Y_gen)..., 
hover = [i for i in 1:lastindex(eachcol(Z_))])

norm(Y_gen-Y_)/norm(Y_)

function Objective6_(x,x_,X_)
    θ = x[1]
    α = x[2]
    rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
    scale = [α 0;0 α]
    x_ = rot*x_
    x_ = scale*x_
    return norm(X_-x_)/norm(X_)
end

Objective6(x) = Objective6_(x,Y_gen,Y_)
x0 = [0.0,1.0]
_min = optimize(Objective6, x0, NelderMead())
θ = _min.minimizer[1]
α = _min.minimizer[2]
rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
scale = [α 0;0 α]
Y_gen = rot*Y_gen
Y_gen = scale*Y_gen

norm(Y_gen-Y_)/norm(Y_)
scatter!(eachrow(Y_gen)..., 
hover = [i for i in 1:lastindex(eachcol(Z_))])


X_test = []
for i in 1:3
    push!(X_test,CSV.File("data/csv/EM_PB_10S_Phi2000_test/EM_PB_10S_$i.csv") |> Tables.matrix)
end
X_test = hcat(X_test...)

conf_test = CSV.File("data/csv/EM_PB_10S_Phi2000_test/Config_N60_EM_PB_10S.csv") |> Tables.matrix


y_gen = VectorSearch(Y_,conf_test[:,10])
y_gen = rot*y_gen
y_gen = scale*y_gen
scatter!([y_gen[1]],[y_gen[2]])
x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,false,8)

plot!(x_gen,label="Generated X_test[10]_2", legend=:bottomright)
plot!(X_test[:,10],label="FOD X_test[10]")

norm(X_test[:,10]-x_gen)/norm(X_test[:,10])

z_gen, w_ns, Z_ns = ReverseMap(Z_,Y_,y_gen,false,8)
x_gen, w_ns, Z_ns = ReverseMap(X,Z_,z_gen,false,8)
scatter!([z_gen[1]],[z_gen[2]],[z_gen[3]])

norm(X_test[:,10]-x_gen)/norm(X_test[:,10])

err1 = [Inf64 for i in 1:30]
err2 = err1
for i in 1:30
    y_gen = VectorSearch(Y_,conf_test[:,i])
    y_gen = rot*y_gen
    y_gen = scale*y_gen
    x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,false,8)
    err1[i] = norm(X_test[:,i]-x_gen)/norm(X_test[:,i])
    z_gen, w_ns, Z_ns = ReverseMap(Z_,Y_,y_gen,false,8)
    x_gen, w_ns, Z_ns = ReverseMap(X,Z_,z_gen,false,8)
    err2[i] = norm(X_test[:,i]-x_gen)/norm(X_test[:,i])
end
scatter(err1)
scatter!(err2)
mean(err1)




conf_complete = []
for x1 in 0:2, x2 in 0:2, x3 in 0:2, x4 in 0:2, x5 in 0:2, x6 in 0:2, x7 in 0:2, x8 in 0:2, x9 in 0:2, x10 in 0:2
  push!(conf_complete,[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
end
conf_complete = reduce(hcat,conf_complete)

Y_gen_complete = []
for c in eachcol(conf_complete)
    y_ = VectorSearch(Y_,c)
    push!(Y_gen_complete,y_)
end
Y_gen_complete = reduce(hcat,Y_gen_complete)
Y_gen_complete = rot*Y_gen_complete
Y_gen_complete = scale*Y_gen_complete

id_conf_list = [0 for i in 1:30]
n_test = 1
for n_test in 1:30
    x_test = X_test[:,n_test]
    y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    kdtree = KDTree(Y_gen_complete; leafsize = 10)
    idxs, dists = knn(kdtree, y_new, 200, true)
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

scatter(id_conf_list)
mean(id_conf_list)

Y_ns = copy(Y_)
new_conf = []
for i in 1:100
    kdtree = KDTree(Y_ns; leafsize = 10)
    dist_list = []
    for y_gen_complete in eachcol(Y_gen_complete)
        idx, dist = nn(kdtree, y_gen_complete)
        push!(dist_list,dist)
    end
    itr_min = argmin(dist_list)
    Y_gen_complete[:,indices[[1:600...]]]
end
indices = sortperm(dist_list,rev=true)
dist_list[indices[[1:20...]]]'

scatter!(eachrow(Y_gen_complete[:,indices[[1:600...]]])...)