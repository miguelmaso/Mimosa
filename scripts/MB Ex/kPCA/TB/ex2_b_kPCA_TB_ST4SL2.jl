
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

# -------------------- Data collection (X and conf_list) ----------------------------------------
X, X_ = ReadData2(2000,4)

conf_list = []
for i in [0,1], j in [0,1], k in [0,1], l in [0,1], m in [0,1], n in [0,1], o in [0,1], p in [0,1]
    push!(conf_list,[i,j,k,l,m,n,o,p])
end
conf_list = reduce(hcat,conf_list)

# ----------------------kPCA ---------------------
k = 3
β = β_min.minimizer # 1000
β = 25.477396507076456
β = 0.01
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Κ(X1,X2) = (X1'*X2 + β)^2
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)


# ----------- kPCA RS scatter coloring a single PP ----------------------------------
param = 8
plotlyjs()
gr()
s = scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3", color = [c[param] for c in conf_list])
display(s)
savefig("data/Figs/kPCA_TBST4SL2/256_kPCAGaussBeta$β-ColorParam$param.png")

# ----------- kPCA RS scatter NO coloring ---------------------------------------
s = scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3", xlims=(minimum(Z_[1,:]),maximum(Z_[1,:])), 
ylims=(minimum(Z_[1,:]),maximum(Z_[1,:])), zlims=(minimum(Z_[1,:]),maximum(Z_[1,:])),
camera = (30 , 55))

# ----------------- kPCA eigen value analysis -----------------------------------
Λ = real.(Λ)
Λ_t = sum(Λ)
Λ = (1/Λ_t)*Λ
Λ_s = round.(100*Λ[[i for i in 1:10]])
p = plot(real.(Λ_s),type="bar",xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
display(p)


# ----------------------- Colletion and plotting of kPCA clusters -------------
r = 1.1*norm(Z_[:,43]-Z_[:,35])
kdtree = KDTree(Z_; leafsize = 10)
idxs = inrange(kdtree,Z_[:,43],r)

r = 1.1*norm(Z_gen[:,198]-Z_[:,205])
kdtree = KDTree(Z_gen; leafsize = 10)
idxs = inrange(kdtree,Z_[:,198],r)

plotlyjs()
gr()
pyplot()
plotSet(idxs,conf_list,X_)
# ---------------------------------------------------------------------------



# --------------------- isomap -------------------------------------------
neighbors = 25
Y_, _ = isomap1(neighbors,Z_)
s = scatter(eachrow(Y_)..., hover = [i for i in 1:lastindex(eachcol(Y_))],
xlabel="y1",ylabel="y2",label="TS")


# -------------------- parameter setup for VS and TS generation in RS-------------------------------

# wrong--------------
VS_Conf = zeros(8,8)
for i in 1:8
    VS_Conf[i,i] = 1.0
end

VS_Conf_list = []
for i in 1:8
    j = 0
    for c in eachcol(conf_list)
        j+=1
        if VS_Conf[:,i]==c
            push!(VS_Conf_list,j)
        end
    end
end

Y_gen = []
for i in 1:lastindex(eachcol(conf_list))
    conf = conf_list[:,i]
    push!(Y_gen,VectorSearch_wrong(Y_,conf,VS_Conf_list))
end

Y_gen = reduce(hcat,Y_gen)
scatter!(eachrow(Y_gen)..., 
hover = [i for i in 1:lastindex(eachcol(Y_))],label="TS gen (wrong)")

# Right ----------------
conf_a = [digits(i-1,base=2,pad=4) for i in 1:16]
conf_a = reduce(hcat,conf_a)
conf_ = vcat(conf_a,zeros(4,16))
conf_ = hcat(conf_,vcat(zeros(4,16),conf_a))



VS_Conf_list = []
for i in 1:32
    j = 0
    for c in eachcol(conf_list)
        j+=1
        if conf_[:,i]==c
            push!(VS_Conf_list,j)
        end
    end
end

Y_gen = []
for i in 1:lastindex(eachcol(conf_list))
    conf = conf_list[:,i]
    push!(Y_gen,VectorSearch(Y_,conf,VS_Conf_list))
end

Y_gen = reduce(hcat,Y_gen)
scatter!(eachrow(Y_gen)..., 
hover = [i for i in 1:lastindex(eachcol(Y_))], label="TS gen")

scatter(eachrow(Y_[:,idxs])..., hover = idxs,
xlabel="z1",ylabel="z2")

scatter!(eachrow(Y_gen[:,idxs])..., hover = idxs,
xlabel="z1",ylabel="z2")

conf_list[:,idxs]
idxs'

norm(Y_gen-Y_)/norm(Y_)

norm(X[:,153]-X[:,158])/norm(X[:,158])

# Quadric surface explicit in z3 of the form z3 = A + B*z1^2 + Cz2^2 + D*z1 + E*z2
Quadric(z) = [1 z[1]^2 z[2]^2 z[1] z[2]] 

# # Necessary when data fits to a surface explicit in z2 instead of z3
# Z_c = Z_[3,:]
# Z_[3,:] = Z_[2,:]
# Z_[2,:] = Z_c

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

# Compute R for the fitted curve
SS_reg = sqrt(dot(Z_[3,:]-z3.(Z_[1,:],Z_[2,:]),Z_[3,:]-z3.(Z_[1,:],Z_[2,:])))
avr = sum(Z_[3,:])/length(Z_[3,:])
SS_tot = sqrt(dot(Z_[3,:].-avr,Z_[3,:].-avr))
R_sq = 1-SS_reg/SS_tot
R = sqrt(R_sq)

Z_gen = []
for conf in eachcol(conf_list)
    push!(Z_gen,VectorSearch(Z_,conf,VS_Conf_list))
end
Z_gen = reduce(hcat,Z_gen)
scatter!(eachrow(Z_gen)..., hover = [i for i in 1:lastindex(eachcol(Y_))])
norm(Z_gen-Z_)/norm(Z_)





Err = objective(83)

β_min = optimize(objective, 0.0, 1500.0, GoldenSection(),abs_tol=1.0e-8)