# --------------------- Ingput data -------------
pot = 2000
parts = 4
X, X_ = ReadData2(pot,parts)
conf_list = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_Conf0.csv") |> Tables.matrix
conf_list = hcat(conf_list,CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix)
n = lastindex(eachcol(X))
conf_list = conf_list[:,[1:n...]]

# ----------------- kPCA ----------------------
k = 3
β = 47.09735445305706
β_min = optimize(Objective, 0.0, 1500.0, GoldenSection(),abs_tol=1.0e-8)
β = β_min.minimizer
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
plotlyjs()
gr()
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3",size=(1000,1000))

# ----------------------- isomap ---------------------
neighbors = 25
Y_, D_G_sym = isomap1(neighbors,Z_)
s = scatter(eachrow(Y_)...,xlabel="y1",ylabel="y2",label="TS",legend=:outerbottom,legend_columns=2,size=(1000,1000))

# ------------- VS TS generation -----------------
VS_Conf_list = [1:64...]
Y_gen = []
for i in 1:lastindex(eachcol(conf_list))
    conf = conf_list[:,i]
    push!(Y_gen,VectorSearch(Y_,conf,VS_Conf_list))
end
Y_gen = reduce(hcat,Y_gen)
scatter!(eachrow(Y_gen)...,xlabel="y1",ylabel="y2",label="TS gen")
Err_TS = norm(Y_gen-Y_)/norm(Y_)

# ---------------------- Complete DS generation --------------------
conf_complete = []
for i1 in 0:1, i2 in 0:1, i3 in 0:1, i4 in 0:1, i5 in 0:1, i6 in 0:1, i7 in 0:1, i8 in 0:1, i9 in 0:1, i10 in 0:1, i11 in 0:1, i12 in 0:1, i13 in 0:1, i14 in 0:1, i15 in 0:1, i16 in 0:1
    push!(conf_complete,[i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16])
end
conf_complete = reduce(hcat,conf_complete)


Y_complete = []
for i in 1:lastindex(eachcol(conf_complete))
    conf = conf_complete[:,i]
    push!(Y_complete,VectorSearch(Y_,conf,VS_Conf_list))
end
Y_complete = reduce(hcat,Y_complete)
scatter!(eachrow(Y_complete)...,xlabel="y1",ylabel="y2")

# ---------------- FOS generation for the complete DS -------------
x_gen_complete = []
for i in 1:lastindex(eachcol(conf_complete))
    print("\r$i")
    y_ = Y_complete[:,i]
    conf_ = conf_complete[:,i]
    push!(x_gen_complete,ReverseMap2(y_,conf_,Y_,X,conf_list,nh))
end
x_gen_complete = reduce(hcat,x_gen_complete)
x_gen_complete = Float64.(x_gen_complete)


# Test used to develop the ReverseMap2 function ------------------
plot_x(X[:,65])

conf = conf_list[:,65]# [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0] # rand(eachcol(conf_complete))
y_gen = VectorSearch(Y_,conf,VS_Conf_list)
x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,5)
plot_x(x_gen)
plot_x!(X[:,2])
X[:,16]
conf'
X

x_gen = ReverseMap2([0.0,0.0000001],[0 for i in 1:16],Y_,X,conf_list,5)
plot_x!(x_gen)
plot(x_gen)

x_gen = ReverseMap2(y_gen,conf,Y_,X,conf_list,5)
x_gen = ReverseMap3(y_gen,conf,Y_,X,conf_list,5)
plot_x!(x_gen)
plot(x_gen)

nh = 1
conf = conf_list[:,65] # rand(eachcol(conf_complete))
y_gen = VectorSearch(Y_,conf,VS_Conf_list)
x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,nh)
plot_x(x_gen)
x_gen = ReverseMap3(y_gen,conf,Y_,X,conf_list,nh)
plot_x!(x_gen)
x_gen = ReverseMap2(y_gen,conf,Y_,X,conf_list,nh)
plot_x!(x_gen)


# Input test data ------------------------------------
_X_test = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_St4_Sl4_100.csv") |> Tables.matrix
_X_test = _X_test'
n = 10
X_test = []
push!(X_test,_X_test[:,[1:n...]])
push!(X_test,_X_test[:,[n+1:2*n...]])
push!(X_test,_X_test[:,[2*n+1:3*n...]])
X_test = reduce(vcat,X_test)

conf_list_test = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
conf_list_test = conf_list_test[:,[1000-9:1000...]]



# ------------ BCRM for single shape and err for test shapes ----------------
nh = 1
n_test = 4
conf = conf_list_test[:,n_test] # rand(eachcol(conf_complete))
y_gen = VectorSearch(Y_,conf,VS_Conf_list)
plot_x(X_test[:,n_test],"FOS TestID=$n_test conf=$conf")
x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,nh)
plot_x!(x_gen,"FOS generated Simple ReverseMap")
x_gen = ReverseMap2(y_gen,conf,Y_,X,conf_list,nh)
plot_x!(x_gen,"FOS generated BCRM")
Err = norm(x_gen-X_test[:,n_test])/norm(X_test[:,n_test])




Err_test = []
for nh in 3:3, n_test in 1:10
    conf = conf_list_test[:,n_test] # rand(eachcol(conf_complete))
    y_gen = VectorSearch(Y_,conf,VS_Conf_list)
    x_gen = ReverseMap2(y_gen,conf,Y_,X,conf_list,nh)
    Err = norm(x_gen-X_test[:,n_test])/norm(X_test[:,n_test])
    push!(Err_test,Err)
end
plot(Err_test,type=:bar,xlabel="Test ID",ylabel="Error",label=false,xticks=[1:10...])

# ------------------ PP Search --------------------------------
n_test = 1
x_test = X_test[:,n_test] 
y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
scatter!([y_new[1]],[y_new[2]],label="ID test = $n_test in RS")
kdtree = KDTree(Y_complete; leafsize = 10)
idxs, dists = knn(kdtree, y_new, 500, true)
Err_ = []
plot_x(X_test[:,n_test])
for idx in idxs
    x_gen = ReverseMap2(Y_complete[:,idx],
    conf_complete[:,idx],Y_,X,conf_list,3)
    # p = plot_x!(x_gen)
    push!(Err_,(norm(x_test-x_gen))/norm(x_test))
    # push!(Err_,Κ(x_test,x_gen))
    print("\ridx running = $idx")
end
display(p)
# plts[n_test] = p
Indices = sortperm(Err_)
scatter!(eachrow(Y_complete[:,idxs])...,label="500 generated nearest neighbors")
id_conf = 0
id_idx = 0
for i in 1:lastindex(Indices)
    if conf_complete[:,idxs[Indices[i]]]==conf_list_test[:,n_test]
        id_conf = i
        id_idx = idxs[Indices[i]]
        println("\n$i")
    end
end
x_gen = ReverseMap2(Y_complete[:,id_idx],
    conf_complete[:,id_idx],Y_,X,conf_list,3)
p = plot_x!(x_gen)

NH = []
E_ = []
p = [plot_x(X_test[:,n_test],"FOS ID test = $n_test") for n_test in 1:10]
for n_test in 1:10
    println(n_test)
    x_test = X_test[:,n_test]
    p[n_test] = plot_x(x_test,"FOS ID test = $n_test")
    y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
    # s = scatter!([y_new[1]],[y_new[2]],label="ID test = $n_test in RS")
    kdtree = KDTree(Y_complete; leafsize = 10)
    o = 500
    idxs, dists = knn(kdtree, y_new, o, true)
    Err_ = []
    for idx in idxs
        x_gen = ReverseMap2(Y_complete[:,idx],
        conf_complete[:,idx],Y_,X,conf_list,3)
        # p = plot_x!(x_gen)
        push!(Err_,(norm(x_test-x_gen))/norm(x_test))
        # push!(Err_,Κ(x_test,x_gen))
        # print("\ridx running = $idx")
    end
    # plts[n_test] = p
    Indices = sortperm(Err_)
    x_gen = ReverseMap2(Y_complete[:,idxs[Indices[1]]], conf_complete[:,idxs[Indices[1]]],Y_,X,conf_list,3)
    p[n_test] = plot_x!(x_gen,"FOSgen of ID test = $n_test")
    # -----------------  FOS via simulation   --------------------------------------------------------------------
    ph, chache = main(; get_parameters(2000.0, conf_complete[:,idxs[Indices[1]]], 4, 4)...)
    x_line = CenterLine(ph)
    x_line_list = [x_line]
    x_line_list = [getproperty.(x_line,:data) for x_line in x_line_list]
    x_line_list = [x_line_list[i][j][k] for i in 1:lastindex(x_line_list), j in 1:lastindex(x_line_list[1]), k in 1:lastindex(x_line_list[1][1])]
    df = DataFrame(vcat(x_line_list[:,:,1],x_line_list[:,:,2],x_line_list[:,:,3]), :auto)
    CSV.write("data/csv/EM_TB_St4_Sl4_test_$n_test.csv", df)
    _X_test = CSV.File("data/csv/EM_TB_St4_Sl4_test_$n_test.csv") |> Tables.matrix
    _X_test = _X_test'
    n = 1
    X_test_gen = []
    push!(X_test_gen,_X_test[:,[1:n...]])
    push!(X_test_gen,_X_test[:,[n+1:2*n...]])
    push!(X_test_gen,_X_test[:,[2*n+1:3*n...]])
    X_test_gen = reduce(vcat,X_test_gen)
    p[n_test] = plot_x!(X_test_gen,"FOS of found PP for ID test = $n_test")
    push!(E_,(norm(x_test-X_test_gen))/norm(x_test))
    # -----------------------------------------------------------------------------------------
    id_conf = 0
    id_idx = 0
    for i in 1:o#lastindex(Indices)
        if conf_complete[:,idxs[Indices[i]]]==conf_list_test[:,n_test]
            id_conf = i
            id_idx = idxs[Indices[i]]
            println("\n$i")
        end
    end
    push!(NH,id_conf)
end
plot(NH,type=:bar)
plot(E_,type=:bar,xlabel="Test ID",ylabel="Error norm/norm",label=false)
display(s)

kdtree = KDTree(x_gen_complete; leafsize = 10)
NH = []
for n_test in 1:10
    idxs, dists = knn(kdtree,X_test[:,n_test],20000,true)
    Count = 1
    nh = 0
    for id in idxs
        if conf_list_test[:,n_test]==conf_complete[:,id]
            nh = Count
        end
        Count += 1
    end
    push!(NH,nh)
end
plot(NH, type=:bar)

plot_x(X_test[:,3])
conf_list_test[:,3]'
y_gen = VectorSearch(Y_,conf_list_test[:,3],VS_Conf_list)
x_gen = ReverseMap2(y_gen,conf_list_test[:,3],Y_,X,conf_list,nh)
plot_x!(x_gen)