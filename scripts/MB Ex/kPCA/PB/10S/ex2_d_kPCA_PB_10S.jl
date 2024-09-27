n_p, P, d = 7, [2000], false
# Training Set in the full-order space
X = ReadData(n_p, P, d)
# Number of principal directions to be considered
k = 3
#Generation of full set of parameters in order
conf = CSV.File("data/csv/EM_PB_10S_Phi2000/Config_N1422_EM_PB_10S.csv") |> Tables.matrix

β_min = optimize(Objective7, 0.0, 1.0, GoldenSection(),abs_tol=1.0e-8)
β = β_min.minimizer
β = 0.15806570088783478 # Smart TS 200 optimization
β = 0.1538588229886658 # Obtained by optimizing the error of the whole procedure to regenerate the TS via Vector Search
β = 0.31
β = 6.1803437071590605e19
β = 1.0e6
for i in 1:3
    print("\r$i")

end

function Der2nd(x,h)
    n = length(x)
    x_d2 = []
    for i in 2:n-1
        d2 = (x[i-1] - 2*x[i] + x[i+1])/(h^2)
        push!(x_d2,d2)
    end
    return(x_d2)
end
gr()
n_test = 30 
plot(X[:,n_test],label="X",title="$(conf[:,n_test])",color=1)
X_d2_1 = Der2nd(X[:,n_test],0.001)
plot!(twinx(),X_d2_1,yaxis=:left,label="D2_X",color=2)

Κ(X1,X2) = (Der2nd(X1,0.001)'*Der2nd(X2,0.001) + β)^2
Κ(X1,X2) = exp(-β*(dot(Der2nd(X1,0.001)-Der2nd(X2,0.001),Der2nd(X1,0.001)-Der2nd(X2,0.001))))
Κ(X[:,1],X[:,1])

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

plotlyjs()
Λ_t = sum(Λ)
Λ = (1/Λ_t)*Λ
Λ_s = round.(100*Λ[[i for i in 1:10]])
p = plot(real.(Λ_s),type=:bar,xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
display(p)
sum(Λ_s)

Σ = diagm(real.(sqrt.(complex(Λ))))
Σ_ = Σ[:,[1:10...]]
U_ = U*pinv(Σ_')
Z_ = real.(U_'*Ḡ)

scatter(conf[10,:],Z_[10,:],xlabel="Conf parameter value",ylabel="z in the corresponding direction")
conf[:,3]'
Z_[:,3]'
X_d2_1 = Der2nd(X[:,1],0.001)
plot(X_d2_1)
Z_ = Z_./norm(Z_)
conf[:,10]'*Z_[:,10]
Z__ = [z./norm(z) for z in eachcol(Z_)]
Z_ = reduce(hcat,Z__)
Z__ = [z./Z_[:,1] for z in eachcol(Z_)]


neighbors = 25
t_o = time()
Y_, D_G_sym = isomap1(neighbors,Z_)
t_i = time() - t_o
scatter(eachrow(Y_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="y1",ylabel="y2", label = "TS")
# color = [col(i) for i in 1:lastindex(eachcol(conf))])

function col(i)
    if i<=622
        return 1
    elseif i<=822
        return 2
    elseif i<=1022
        return 3
    else
        return 4
    end
end

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

Err = []
Time = []
for nh in 29:35
    println("nh in the works $nh")
    t_o = time()
    neighbors = nh
    Y_, D_G_sym = isomap1(neighbors,Z_)
    n = lastindex(eachcol(Y_))
    Y_gen = []
    for i in 1:n
        y_ = VectorSearch(Y_,conf[:,i])
        push!(Y_gen,y_)
    end
    Y_gen = reduce(hcat,Y_gen)
    e = norm(Y_gen-Y_)/norm(Y_)
    t_f = time() - t_o
    println("Error: $e")
    println("Elapsed Time: $t_f")
    push!(Err, e)
    push!(Time, t_f)
end
12
plot(Err)
plot!(twinx(),Time)
Data = hcat(nh,Err,Time)
df = DataFrame(Data', :auto)
CSV.write("data/csv/Err&TimeVSNumOfNeighborsForIsomap_1022.csv", df)

Err_0 = CSV.File("data/csv/Err&TimeVSNumOfNeighborsForIsomap_1022.csv") |> Tables.matrix
plot([Err_0[1,:]],[Err_0[2,:]])
plot((),[Err_0[1,:]],[Err_0[3,:]])

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

Y_copy = copy(Y_)
param1 = []
for i in 1:n
    push!(param1,conf[1,i])
end
sort_param1 = sortperm(param1)
group_param = 0
group = [[],[],[]]
j=1
for i in 1:n
    if param1[sort_param1[i]]==group_param
        push!(group[j],sort_param1[i])
    else
        j += 1
        group_param += 1
        push!(group[j],sort_param1[i])
    end
end
scatter(eachrow(Y_copy[:,group[1]])..., hover = group[1])
scatter!(eachrow(Y_copy[:,group[2]])..., hover = group[2])
scatter!(eachrow(Y_copy[:,group[3]])..., hover = group[3])

param1_test = conf_test[1,10]
param1_test += 1
x_gen, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]],Y_copy[:,group[param1_test]],y_gen,false,3)
plot(X_test[:,10],label="FOD X_test[10]", legend=:bottomright)
plot!(x_gen,label="Generated X_test[10]_2")
scatter(eachrow(Y_copy[:,group[param1_test]])..., hover = group[3])
scatter!([y_gen[1]],[y_gen[2]])

y_gen = VectorSearch(Y_,conf_test[:,10])
y_gen = rot*y_gen
y_gen = scale*y_gen
# scatter!([y_gen[1]],[y_gen[2]])
x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,false,1)

n_sec, sw, pot,init_sol_bool,init_sol_FOS = 10, conf_test[:,10], 2000.0, true, x_gen
init_sol_bool = false
ph, chache = main(; get_parameters(n_sec, sw, pot,init_sol_bool,init_sol_FOS)...)
init_sol_bool = true
init_sol_FOS = ph[1]
ph, chache = main(; get_parameters(n_sec, sw, pot,init_sol_bool,init_sol_FOS)...)


Poly(z) = [1 z z^2 z^3 z^4 z^5 z^6 z^7 z^8 z^9 z^9] 
A = [Poly(x) for x in 0.0:0.001:0.1]
A = reduce(vcat,A)
c = pinv(A)*x_gen
x_gen2 = A*c
function U_ap(X)
    x_3 = Poly(X[1])*c
    return Gridap.VectorValue([0.0,0.0,x_3[1]-0.0004])
end
function X_ap(X)
    return 0.0
end
using GridapGmsh
model = GmshDiscreteModel("data/models/PlateBeam10SecSI.msh")
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},2)
V = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags = ["fixedup_1"])

reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, 2)
reffeφ = ReferenceFE(lagrangian, Float64, 2)

# Test FE Spaces
Vu = TestFESpace(model, reffeu, dirichlet_tags= ["fixedup_1"], conformity=:H1)
Vφ = TestFESpace(model, reffeφ, dirichlet_tags= dir_φ_tags, conformity=:H1)

# Trial FE Spaces
Uu = TrialFESpace(Vu, [g])
Uφ = TrialFESpace(Vφ, dir_φ_values)

V = MultiFieldFESpace([Vu, Vφ])
U = MultiFieldFESpace([Uu, Uφ])

g(x) = VectorValue([0.0, 0.0, 0.0])
# U = TrialFESpace(V,[g])
f = interpolate([U_ap,X_ap],U)
writevtk(Ω,"results_MB",cellfields=["uh"=>f[1], "phi"=>f[2]])
x0 = zeros(Float64,num_free_dofs(V))
uh = FEFunction(U,1)


plot!(x_gen,label="Generated X_test[10]_1", legend=:bottomright)
plot(X_test[:,10],label="FOD X_test[10]")

norm(X_test[:,10]-x_gen)/norm(X_test[:,10])

z_gen, w_ns, Z_ns = ReverseMap(Z_,Y_,y_gen,false,8)
x_gen, w_ns, Z_ns = ReverseMap(X,Z_,z_gen,false,8)
scatter!([z_gen[1]],[z_gen[2]],[z_gen[3]])

norm(X_test[:,10]-x_gen)/norm(X_test[:,10])

err1 = [Inf64 for i in 1:30]
err2 = err1
for i in 1:30
    param1_test = conf_test[1,i]
    param1_test += 1
    y_gen = VectorSearch(Y_,conf_test[:,i])
    # y_gen = rot*y_gen
    # y_gen = scale*y_gen
    x_gen, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]],Y_[:,group[param1_test]],y_gen,false,8)
    err1[i] = norm(X_test[:,i]-x_gen)/norm(X_test[:,i])
    z_gen, w_ns, Z_ns = ReverseMap(Z_[:,group[param1_test]],Y_[:,group[param1_test]],y_gen,false,8)
    x_gen, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]],Z_[:,group[param1_test]],z_gen,false,8)
    err2[i] = norm(X_test[:,i]-x_gen)/norm(X_test[:,i])
end
scatter(err1)
plot(err1, type=:bar, ylims=(0.0,maximum(err1)),legend=false,
ylabel="ReverseMap Err (norm/norm)",xlabel="Test Conf ID")
scatter!(err2)
mean(err1)

err = []
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
    err1 = [Inf64 for i in 1:30]
    for i in 1:30
        param1_test = conf_test[1,i]
        param1_test += 1
        y_gen = VectorSearch(Y_,conf_test[:,i])
        # y_gen = rot*y_gen
        # y_gen = scale*y_gen
        x_gen, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]],Y_[:,group[param1_test]],y_gen,false,8)
        err1[i] = norm(X_test[:,i]-x_gen)/norm(X_test[:,i])
    end
    push!(err,mean(err1))
end
plot(err, type=:bar, ylims=(0.0,0.3),legend=false,
ylabel=" Average RM Err (norm/norm) Over the 30 test Conf",xlabel="Param ID")

err_list = []
for nh in 1:20
    err1 = [Inf64 for i in 1:30]
    for i in 1:30
        y_gen = VectorSearch(Y_,conf_test[:,i])
        y_gen = rot*y_gen
        y_gen = scale*y_gen
        x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,false,nh)
        err1[i] = norm(X_test[:,i]-x_gen)/norm(X_test[:,i])
    end
    push!(err_list,mean(err1))
end
push!(err,mean(err1))
plot(err_list, type=:bar, ylims=(0.0,0.3))

err1 = [Inf64 for i in 1:30]
y_gen_list = []
for i in 1:30
    y_gen = VectorSearch(Y_,conf_test[:,i])
    y_gen = rot*y_gen
    y_gen = scale*y_gen
    push!(y_gen_list,y_gen)
    x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,false,8)
    err1[i] = norm(X_test[:,i]-x_gen)/norm(X_test[:,i])
end
plot(err1, type=:bar, ylims=(0.0,0.3))
sum(err1)/30.0
y_gen_list = reduce(hcat,y_gen_list)
scatter(eachrow(Y_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="y1",ylabel="y2", label = "TS")
scatter!(eachrow(y_gen_list)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="y1",ylabel="y2", label = "y_gen_list_test")




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


n_test = 10
x_test = X_test[:,n_test]
y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
kdtree = KDTree(Y_gen_complete; leafsize = 10)
idx, dist = nn(kdtree, y_new)
conf_test_i = conf_complete[:,idx]

n_sec = 10
pot = 2000.0
ph, chache = main(; get_parameters(n_sec, conf_test_i, pot)...)
pl_, x_FOS = plt_cl(ph,conf_test_i)

plot!(x_FOS,label="Generated", legend=:bottomright)
plot(X_test[:,10],label="FOD X_test[10]")

norm(x_test-x_FOS)/norm(x_test)

Err = [0.0 for i in 1:30]
for n_test in 1:30
    x_test = X_test[:,n_test]
    y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    kdtree = KDTree(Y_gen_complete; leafsize = 10)
    idx, dist = nn(kdtree, y_new)
    conf_test_i = conf_complete[:,idx]
    ph, chache = main(; get_parameters(10, conf_test_i, 2000.0)...)
    pl_, x_FOS = plt_cl(ph,conf_test_i)
    Err[n_test] = norm(x_test-x_FOS)/norm(x_test)
end


plot(Err, type=:bar, ylabel="Err norm(x_test-x_FOS_nearestGenPt)/norm(x_test)",
xlabel="Test configuration ID", label="Err")
MeanError = Statistics.mean(Err)
MeanErr = [MeanError for i in 1:30]
ME = round(MeanError, digits=4)
plot!(MeanErr, linestyle=:dash, linewidth=4, label="Mean Err = $ME")
savefig("data/Figs/DifferenceOver30TestSearchofPP_SmartTS600.png")

y_new_list = []
idx_list = []
dist_list = []
for n_test in 1:30
    x_test = X_test[:,n_test]
    y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    kdtree = KDTree(Y_gen_complete; leafsize = 10)
    idx, dist = nn(kdtree, y_new)
    conf_test_i = conf_complete[:,idx]
    push!(y_new_list,y_new)
    push!(idx_list,idx)
    push!(dist_list,dist)
end

y_new_list = reduce(hcat,y_new_list)
scatter(Err)
scatter!(1e3*dist_list)
scatter(eachrow(Y_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="y1",ylabel="y2", label = "TS")
scatter(eachrow(Y_gen_complete)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="y1",ylabel="y2", label = "TS")
scatter!(eachrow(Y_gen_complete[:,idx_list])..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="y1",ylabel="y2", label = "TS")
scatter!([y_new[1]],[y_new[2]])
scatter!(eachrow(y_new_list)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
label = "y_new_test")
scatter!(eachrow(y_new_list[:,[20,27]])..., hover = [i for i in 1:lastindex(eachcol(Z_))],
label = "y_new_test")

id_conf_list = [0 for i in 1:30]
for n_test in 1:30
    x_test = X_test[:,n_test]
    y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    kdtree = KDTree(Y_gen_complete; leafsize = 10)
    idxs, dists = knn(kdtree, y_new, 300, true)
    id_conf = 0
    for i in 1:lastindex(idxs)
        if conf_complete[:,idxs[i]]==conf_test[:,n_test]
            id_conf = i
            println(i)
        end
    end
    # conf_complete[:,idxs[id_conf]]
    # conf_test[:,n_test]
    id_conf_list[n_test] = id_conf
end

plot(id_conf_list, type=:bar, ylabel="cloest neighbor with the correct configuration", xlabel="Test configuration ID")
mean(id_conf_list)

Err = []
# plts = []
tol = 1e-8
for n_test in 1:30
    x_test = X_test[:,n_test]
    p = plot(X_test[:,n_test],label="Test FOS $(conf_test[:,n_test])")
    y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    kdtree = KDTree(Y_gen_complete; leafsize = 10)
    idxs, dists = knn(kdtree, y_new, 100, true)
    conf_test_i = conf_complete[:,idxs[1]]
    println("n_test simulation running = $n_test at idx 1")
    ph, chache = main(; get_parameters(10, conf_test_i, 2000.0)...)
    pl_, x_FOS = plt_cl(ph,conf_test_i)
    E_ = norm(x_test-x_FOS)/norm(x_test)
    push!(Err,E_)
    println("Error found at nh 1 = $E_")
    if E_ < tol
        push!(Err,E_)
        println("Error was under tolerance $tol")
    else
        Err_ = []
        println("Error was not under tolerance")
        for idx in idxs
            x_gen = ReverseMap2(Y_gen_complete[:,idx],
            conf_complete[:,idx],Y_,X,conf,5,0.0)
            p = plot!(x_gen,label="idx: $idx FOS $(conf_complete[:,idx])")
            push!(Err_,norm(x_test-x_gen)/norm(x_test))
            # push!(Err_,Κ(x_test,x_gen))
            print("\ridx running = $idx")
        end
        index = argmin(Err_)
        if idxs[index]≠idxs[1]
            println("New nh found via RM")
            println("$(idxs[index]) - $(idxs[1])")
            # push!(Err,Err_)
            # push!(plts,p)
            conf_test_i = conf_complete[:,idxs[index]]
            println("n_test simulation running = $n_test")
            ph, chache = main(; get_parameters(10, conf_test_i, 2000.0)...)
            pl_, x_FOS = plt_cl(ph,conf_test_i)
            push!(Err,norm(x_test-x_FOS)/norm(x_test))
            println("Error found at new nh = $(norm(x_test-x_FOS)/norm(x_test))")
        else
            println("No new nh found")
            push!(Err,E_)
        end
    end
end
# h = 1
plot(Err, type=:bar)
mean(Err)
# plot(plts[h],legend=:outerbottom)
# argmin(Err[h])
plot(Err2, type=:bar, ylabel="Err norm(x_test-x_FOS_GenPt)/norm(x_test)",
xlabel="Test configuration ID", label="Err")
MeanError = Statistics.mean(Err2)
MeanErr = [MeanError for i in 1:30]
ME = round(MeanError, digits=4)
plot!(MeanErr, linestyle=:dash, linewidth=4, label="Mean Err = $ME")
savefig("data/Figs/DifferenceOver30TestSearchofPP_SmartTS800_ImprovedRM2.png")
ERR = Err # Error with improved reversemap 1 and increased TS
ERR2 = Err # Error with just increased TS
Err1 = []
Err2 = []
for i in 1:lastindex(Err)
    if i/2 == round(i/2)
        push!(Err2,Err[i])
    else
        push!(Err1,Err[i])
    end
end






NH = [0 for i in 1:30]
plts = [0 for i in 1:30]
ids_idx = [0 for i in 1:30]
idx_1 = [0 for i in 1:30];
Threads.@threads for n_test in 1:30
    x_test = X_test[:,n_test]
    # p = plot(X_test[:,n_test],label="Test FOS $(conf_test[:,n_test])")
    y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    kdtree = KDTree(Y_gen_complete; leafsize = 10)
    idxs, dists = knn(kdtree, y_new, 100, true)
    idx_1[n_test] = idxs[1]
    Err_ = []
    for idx in idxs
        x_gen = ReverseMap2(Y_gen_complete[:,idx],
        conf_complete[:,idx],Y_,X,conf,5,0.0)
        # p = plot!(x_gen,label="idx: $idx FOS $(conf_complete[:,idx])")
        Δ_ = x_test-x_gen
        # f = 0
        # for k in 1:lastindex(Δ_)-1
        #     if Δ_[k]*Δ_[k+1]<0
        #         f = 10
        #     else
        #         f = 1
        #     end
        # end
        push!(Err_,(norm(x_test-x_gen))/norm(x_test))
        # push!(Err_,Κ(x_test,x_gen))
        print("\ridx running = $idx")
    end
    # plts[n_test] = p
    Indices = sortperm(Err_)
    id_conf = 0
    id_idx = 0
    for i in 1:lastindex(Indices)
        if conf_complete[:,idxs[Indices[i]]]==conf_test[:,n_test]
            id_conf = i
            id_idx = idxs[Indices[i]]
            println("\n$i")
        end
    end
    NH[n_test] = id_conf
    ids_idx[n_test] =id_idx
end
plot(NH,type=:bar)
display(plts[30])
ids_idx[30]
NH[30]
conf_complete[:,idx_1[30]]















Y_ns = copy(Y_)
new_conf = []
kdtree = KDTree(Y_ns; leafsize = 10)
dist_list = []
for y_gen_complete in eachcol(Y_gen_complete)
    idx, dist = nn(kdtree, y_gen_complete)
    push!(dist_list,dist)
end
indices = sortperm(dist_list)
Y_gen_complete[:,indices[[1:600...]]]
indices = sortperm(dist_list,rev=true)
dist_list[indices[[1:20...]]]'
nh = 100
scatter!(eachrow(Y_gen_complete[:,indices[[1:nh...]]])..., label = "$nh")

Y_ns = copy(Y_)
new_conf_index = []
n_ = 800
for i in 1:n_
    kdtree = KDTree(Y_ns; leafsize = 10)
    dist_list = []
    for y_gen_complete in eachcol(Y_gen_complete)
        idx, dist = nn(kdtree, y_gen_complete)
        push!(dist_list,dist)
    end
    itr_min = argmax(dist_list)
    push!(new_conf_index,itr_min)
    Y_ns = hcat(Y_ns,Y_gen_complete[:,itr_min])
end

scatter(eachrow(Y_gen_complete[:,new_conf_index])..., label="$n_ adding found points to TS")
conf_complete[:,new_conf_index]

df = DataFrame(conf_complete[:,new_conf_index], :auto)
CSV.write("data/csv/Config_N60_EM_PB_10S_Smart800_1.csv", df)