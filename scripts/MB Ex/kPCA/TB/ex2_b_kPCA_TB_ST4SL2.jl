
y_min = minimum(X_[2])
y_max = maximum(X_[2])
z_min = minimum(X_[3])
z_max = maximum(X_[3])
conf = 121; Cc = conf_list[conf]; plot(X_[1][:,conf],X_[2][:,conf],X_[3][:,conf], xlims =(0.0,0.11),
 ylims=(y_min,y_max), zlims=(z_min,z_max),size=(600,600),
 label="Conf:$conf : $Cc", linewidth=6)
conf = 121; Cc = conf_list[conf]; plot!(X_[1][:,conf],X_[2][:,conf],X_[3][:,conf], xlims =(0.0,0.11),
 ylims=(y_min,y_max), zlims=(z_min,z_max),size=(600,600),
 label="Conf:$conf : $Cc", linewidth=6)
#  savefig("data/Figs/kPCA_TBST4SL2/Centerline_$conf.png")


X, X_ = ReadData2(2000,4)

k = 3
β = 1000
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
# Κ(X1,X2) = (X1'*X2 + β)^2
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
conf_list = []
for i in [0,1], j in [0,1], k in [0,1], l in [0,1], m in [0,1], n in [0,1], o in [0,1], p in [0,1]
    push!(conf_list,[i,j,k,l,m,n,o,p])
end
param = 8
plotlyjs()
gr()
s = scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3", color = [c[param] for c in conf_list])
display(s)
savefig("data/Figs/kPCA_TBST4SL2/256_kPCAGaussBeta$β-ColorParam$param.png")

s = scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3", xlims=(minimum(Z_[1,:]),maximum(Z_[1,:])), 
ylims=(minimum(Z_[1,:]),maximum(Z_[1,:])), zlims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
camera = (30 , 55))

Λ = real.(Λ)
Λ_t = sum(Λ)
Λ = (1/Λ_t)*Λ
Λ_s = round.(100*Λ[[i for i in 1:10]])
p = plot(real.(Λ_s),type="bar",xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
display(p)

r = 1.05*norm(Z_[:,145]-Z_[:,151])
kdtree = KDTree(Z_; leafsize = 10)
idxs = inrange(kdtree,Z_[:,145],r)


plotlyjs()
gr()
pyplot()
plotSet(idxs,conf_list,X_)

neighbors = 25
Y_, _ = isomap1(neighbors,Z_)
s = scatter(eachrow(Y_)..., hover = [i for i in 1:lastindex(eachcol(Y_))],
xlabel="z1",ylabel="z2")