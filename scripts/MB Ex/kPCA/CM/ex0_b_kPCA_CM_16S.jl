conf_list = []
for i in 0:2, j in 0:2, k in 0:2, l in 0:2
  push!(conf_list,[i,j,k,l,i,j,k,l,i,j,k,l,i,j,k,l])
end
pot = 3200
X = []
conf_NF_list = []
for conf in conf_list
    try
        push!(X,ReadData_i(conf,pot))
    catch
        push!(conf_NF_list,conf)
    end
end
X = reduce(hcat,X)


k = 3
β = 5
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
plotlyjs()
plot_eigen(Λ)
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3",hover = [i for i in 1:lastindex(eachcol(Z_))])
neighbors = 20
t_o = time()
Y_, D_G_sym = isomap1(neighbors,Z_)
t_i = time() - t_o
scatter(eachrow(Y_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="y1",ylabel="y2", label = "TS")