Dif(Z_,601,610,611,622)

n_p, P, d = 6, [2000], false
# Training Set in the full-order space
X = ReadData(n_p, P, d)
# Number of principal directions to be considered
k = 3
#Generation of full set of parameters in order
conf = CSV.File("data/csv/EM_PB_10S_Phi2000/Config_N622_EM_PB_10S.csv") |> Tables.matrix

#Gaussian Kernel
Κ(X1,X2) = exp(-3.0*(dot(X1-X2,X1-X2)))
# Polynomial Kernel
Κ(X1,X2) = (X1'*X2 + 0.31)^2
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)

# Training Set in the reduced space
Z_ = real.(U_'*Ḡ)

#Generation of full set of parameters in order
conf = CSV.File("data/csv/EM_PB_10S_Phi2000/Config_N622_EM_PB_10S.csv") |> Tables.matrix

# Scatter plot of the TS in RS - Colored parameters
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))], markercolor = [c[2] for c in eachcol(conf)])

# Scatter plot of the TS in RS
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))], 
title="n_p, P = 622, [2000] <br> Κ(X1,X2) = exp(-3.0*(dot(X1-X2,X1-X2)))")
savefig("data/Figs/kPCA_PB_S10_Phi2000/RS_0.png")




# ---------kPCA & isomap -------------------

β_min = optimize(Objective2, 0.00, 0.5, GoldenSection())
β = β_min.minimizer
β = 0.3
# Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Κ(X1,X2) = (X1'*X2 + β)^2
Λ, U, U_, Ḡ = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))],
xlims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
ylims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
zlims=(minimum(Z_[1,:]),maximum(Z_[1,:])))
kdtree = KDTree(Z_; leafsize = 1)
data = Z_
k0 = 1
i = 600
j = 274
k_n = 60
DistG_WPlot(i,j,k_n,k0,data,kdtree)
ylims!(minimum(Z_[1,:]),maximum(Z_[1,:]))


scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))])
scatter!(eachrow(Z_[:,[i,j]])...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))])

DistG(i,j,k_n,k0,data,kdtree)
k1 = 2
Λ, U, G = isomap0(DistG,data,kdtree,k1,k_n)


G = Z_*Z_'
m, n = size(G)
H = I - (1/n)*ones(n,n)
G = 0.5*H'*G*H
f(λ)=-abs(λ)
Λ, U = eigen(G,sortby=f)


m, n = size(Z_)
G = [DistG(i,j,k_n,1,Z_,kdtree) for i in 1:n, j in 1:n]
H = I - (1/n)*ones(n,n)
G_ = 0.5*H'*G*H
f(λ)=-abs(λ)
Λ, U = eigen(G,sortby=f)




Z_1 = U[:,[1,2]]'*Z_
scatter(eachrow(Z_1)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))],
xlims=(minimum(data[1,:]),maximum(data[1,:])),
ylims=(minimum(data[1,:]),maximum(data[1,:])),
zlims=(minimum(data[1,:]),maximum(data[1,:])))

Σ = diagm(real.(sqrt.(abs.(Λ))))
Σ_ = Σ[:,[1:2...]]
U_ = real.(pinv(Σ_)*U)
Z_ = real.(U_*G)


z_ = []
for c in eachcol(conf)
    z = VectorSearch(Z_1,c); z = real.(z)
    push!(z_,z)
end
z_ = reduce(hcat,z_)

norm(z_-Z_1)/norm(Z_1)

scatter(eachrow(data)...,xlabel="z1",ylabel="z2",zlabel="z3",  
title="",hover= [i for i in 1:lastindex(eachcol(Z_))],
xlims=(minimum(data[1,:]),maximum(data[1,:])),
ylims=(minimum(data[1,:]),maximum(data[1,:])),
zlims=(minimum(data[1,:]),maximum(data[1,:])),)

scatter!(eachrow(data[:,[194]])...,xlabel="z1",ylabel="z2",zlabel="z3",  
title="",hover= [i for i in 1:lastindex(eachcol(Z_))])
scatter!(eachrow(Z_[:,[600]])...,xlabel="z1",ylabel="z2",zlabel="z3",  
title="",hover= [i for i in 1:lastindex(eachcol(Z_))])

scatter(eachrow(Z_[:,[601:622...]])...,xlabel="z1",ylabel="z2",zlabel="z3",  
title="",hover= [i for i in 1:lastindex(eachcol(Z_))])
scatter!(eachrow(z_[:,[601:622...]])...,xlabel="z1",ylabel="z2",zlabel="z3",  
title="",hover= [i for i in 1:lastindex(eachcol(z_))])

scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3",  
title="",hover= [i for i in 1:lastindex(eachcol(Z_))])
scatter!(eachrow(z_)...,xlabel="z1",ylabel="z2",zlabel="z3",  
title="",hover= [i for i in 1:lastindex(eachcol(z_))])


conf_complete = []
for x1 in 0:2, x2 in 0:2, x3 in 0:2, x4 in 0:2, x5 in 0:2, x6 in 0:2, x7 in 0:2, x8 in 0:2, x9 in 0:2, x10 in 0:2
  push!(conf_complete,[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
end
conf_complete = reduce(hcat,conf_complete)

z_complete = []
for c in eachcol(conf_complete)
    z = VectorSearch(Z_,c); z = real.(z)
    push!(z_complete,z)
end
z_complete = reduce(hcat,z_complete)

scatter!(eachrow(z_complete)...,xlabel="z1",ylabel="z2",zlabel="z3",  
title="",hover= [i for i in 1:lastindex(eachcol(z_complete))])

β_min = optimize(Objective3, 0.00, 0.5, GoldenSection())
β = β_min.minimizer







# Plot normalized aigen values
Λ_t = sum(Λ)
Λ = (1/Λ_t)*Λ
Λ_s = round.(100*Λ[[i for i in 1:10]])
p = plot(real.(Λ_s),type="bar",xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
display(p)

# ------ Optimization to set a kernel parameter ------
β_min = optimize(Objective2, 0.00, 0.5, GoldenSection())
β = β_min.minimizer
# Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Κ(X1,X2) = (X1'*X2 + β)^2
Λ, U, U_, Ḡ = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
Dif(Z_,601,610,611,622)
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
    hover= [i for i in 1:lastindex(eachcol(Z_))], 
    title=" Κ(X1,X2) = (X1'*X2 + β)^2<br>β = $β")
savefig("data/Figs/kPCA_PB_S10_Phi2000/RS_Polynomial_Optimized.png")
scatter(eachrow(Z_[:,[601,610,611,622]])...)
DDD = []
space = range(start=0.0, step=1e-2, stop=1.0)
for b in space
    push!(DDD,Objective2(b))
end
plot(space,DDD, 
title="err,norm(Z_-z_)/norm(Z_) vs β <br> Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))")
savefig("data/Figs/kPCA_PB_S10_Phi2000/ObjVsBeta_Gaussiano.png")

# -------Curve fitting------
# Quadric surface explicit in z3 of the form z3 = A + B*z1^2 + Cz2^2 + D*z1 + E*z2
Quadric(z) = [1 z[1]^2 z[2]^2 z[1] z[2]] 

# Necessary when data fits to a surface explicit in z2 instead of z3
Z_c = Z_[3,:]
Z_[3,:] = Z_[2,:]
Z_[2,:] = Z_c

# least squares fit of the curve to the TS in the RS
A = [Quadric(Z) for Z in eachcol(Z_)]
A = reduce(vcat,A)
c = pinv(A)*Z_[3,:]

#Definition of surface and unit gradient
z3(z1,z2) = c[1] + c[2]*z1^2 + c[3]*z2^2 + c[4]*z1 + c[5]*z2
∇zᵤ(z1,z2) = (1/norm([2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]))*[2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]

# Plot of data and surface
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))])
z1 = range(minimum(Z_[1,:]), stop=maximum(Z_[1,:]), length=100)
z2 = range(minimum(Z_[2,:]), stop=maximum(Z_[2,:]), length=100)
surface!(z1,z2,z3)
savefig("data/Figs/kPCA_PB_S10_Phi2000/RS_Polynomial_Optimized_SurfFit.png")

# Compute R for the fitted curve
SS_reg = sqrt(dot(Z_[3,:]-z3.(Z_[1,:],Z_[2,:]),Z_[3,:]-z3.(Z_[1,:],Z_[2,:])))
avr = sum(Z_[3,:])/length(Z_[3,:])
SS_tot = sqrt(dot(Z_[3,:].-avr,Z_[3,:].-avr))
R_sq = 1-SS_reg/SS_tot
R = sqrt(R_sq)

# Whole RS found using VectorSearch and considering ∇zᵤ
z_ = []
for c in eachcol(conf)
    z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
    push!(z_,z)
end
z_ = reduce(hcat,z_)
scatter!(eachrow(z_)..., hover= [i for i in 1:lastindex(eachcol(Z_))])
savefig("data/Figs/kPCA_PB_S10_Phi2000/RS_Polynomial_Optimized_SurfFit_Zgen.png")

# evaluation of the distributted error - Applicable if using the whole RS as the TS
norm(Z_-z_)/norm(Z_)

scatter(eachrow(Z_[:,[600:605...]])..., hover= [i for i in 600:lastindex(eachcol(Z_))])
scatter!(eachrow(z_[:,[600:605...]])..., hover= [i for i in 600:lastindex(eachcol(Z_))])

#ReverseMap calculation of configurations not in the TS
x, w_ns, Z_ns = ReverseMap(Z_,z_[:,6],d,8)
plot(x)
plot!(X[:,6])
savefig("data/Figs/kPCA_PB_S10_Phi2000/ReverseMap_X6inTS_8Vecino.png")
conf[:,6]
norm(x-X[:,6])/norm(X[:,6])

e = []
for j in 1:100
    err = []
    for i in 1:600
        x, w_ns, Z_ns = ReverseMap(Z_,z_[:,i],d,j)
        push!(err,norm(x-X[:,i])/norm(X[:,i]))
    end
    push!(e,Statistics.mean(err))
end
plot(e)
savefig("data/Figs/kPCA_PB_S10_Phi2000/ReverseMap_ErrInTSVSns.png")

X_test = []
for i in 1:3
    push!(X_test,CSV.File("data/csv/EM_PB_10S_Phi2000_test/EM_PB_10S_$i.csv") |> Tables.matrix)
end
X_test = hcat(X_test...)

conf_test = CSV.File("data/csv/EM_PB_10S_Phi2000_test/Config_N60_EM_PB_10S.csv") |> Tables.matrix







x_test = X_test[:,10]
Κ_test(X) = Κ(x_test,X)
g_i = Κ_test.(eachcol(X))
G_i = hcat(Ḡ,g_i)
n, m = size(G_i)
II = ones(n,m)
Ḡ_i = G_i - hcat((1/n)*G_i*II',zeros(622,1)) - ((1/n)*II'*G_i)[[1:622...],:] + ((1/n^2)*II'*G_i*II')'
Z_i = real.(U_'*G_i)
norm(Z_-Z_i[:,[1:622...]])

scatter(eachrow(Z_i)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_i))],
xlims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
ylims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
zlims=(minimum(Z_[1,:]),maximum(Z_[1,:])))
scatter!(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))],
xlims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
ylims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
zlims=(minimum(Z_[1,:]),maximum(Z_[1,:])))

z_test = []
for c in eachcol(conf_test)
    z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
    push!(z_test,z)
end
z_test = reduce(hcat,z_test[[1:30...]])
scatter(eachrow(z_test)..., hover= [i for i in 1:lastindex(eachcol(z_test))])
savefig("data/Figs/kPCA_PB_S10_Phi2000/TestSet30_inRS_Optimized.png")

Λ_test, U_test, U__test, Ḡ_test = kPOD(Κ, X_test, k)
Z_test = real.(U_'*Ḡ_test)

n_test = 10
x, w_ns, Z_ns = ReverseMap(Z_,z_test[:,n_test],d,3)
plot(x)
plot!(X_test[:,n_test])
savefig("data/Figs/kPCA_PB_S10_Phi2000/ReverseMap_Xtest10_3Vecinos.png")
conf[:,n_test]
norm(x-X_test[:,n_test])/norm(X_test[:,n_test])

e = []
for j in 1:100
    err = []
    for i in 1:30
        x, w_ns, Z_ns = ReverseMap(Z_,z_test[:,i],d,j)
        push!(err,norm(x-X_test[:,i])/norm(X_test[:,i]))
    end
    push!(e,Statistics.mean(err))
end
plot(e)
savefig("data/Figs/kPCA_PB_S10_Phi2000/ReverseMap_ErrInTestSetVSns.png")

X_test

conf_complete = []
for x1 in 0:2, x2 in 0:2, x3 in 0:2, x4 in 0:2, x5 in 0:2, x6 in 0:2, x7 in 0:2, x8 in 0:2, x9 in 0:2, x10 in 0:2
  push!(conf_complete,[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
end
conf_complete = reduce(hcat,conf_complete)

z_complete = []
for c in eachcol(conf_complete)
    z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
    push!(z_complete,z)
end
z_complete = reduce(hcat,z_complete)

kdtree = KDTree(z_complete; leafsize = 1)
idxs, dists = knn(kdtree, Z_i[:,623], 100, true)
zz = Z_i[:,623]
zz[3] = z3(zz[1],zz[2])
scatter!([zz[1]],[zz[2]],[zz[3]],xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))])
zz
idxs, dists = knn(kdtree, zz, 100, true)

count = 1
for idx in idxs
    if conf_complete[:,idx]==conf_test[:,10]
        println("conf $idx")
        println(count)
    end
    count += 1
end
count


scatter!(eachrow(z_complete[:,idxs])...,xlabel="z1",ylabel="z2",zlabel="z3",
hover= idxs)

scatter!(eachrow(z_complete)...,xlabel="z1",ylabel="z2",zlabel="z3")
savefig("data/Figs/kPCA_PB_S10_Phi2000/CompleteSet_inRS_Optimized.png")

X_complete = []
for z in eachcol(z_complete)
    x, w_ns, Z_ns = ReverseMap(Z_,z,d,3)
    push!(X_complete, x)
end

n_test = 10
Compare = []
for i in 1:lastindex(X_complete)
    push!(Compare,norm(X_complete[i]-X_test[:,n_test])/norm(X_test[:,n_test]))
end
minimum(Compare)
Compare_sort = sortperm(Compare, rev=false)
conf_test[:,10]
conf_complete[:,Compare_sort[1]]
for i in 1:lastindex(Compare_sort)
    if conf_test[:,10]==conf_complete[:,Compare_sort[i]]
        println(i)
    end
end
Compare[Compare_sort[23]]
plot!(X_complete[Compare_sort[23]])
plot(X_test[:,n_test])
x, w_ns, Z_ns = ReverseMap(Z_,z_test[:,n_test],d,3)
plot!(x)


n_test = 10
Compare = []
x_test, w_ns, Z_ns = ReverseMap(Z_,z_test[:,n_test],d,10)
for i in 1:lastindex(X_complete)
    push!(Compare,norm(X_complete[i]-x_test)/norm(x_test))
end
minimum(Compare)
Compare_sort = sortperm(Compare, rev=false)
conf_test[:,10]
conf_complete[:,Compare_sort[1]]

