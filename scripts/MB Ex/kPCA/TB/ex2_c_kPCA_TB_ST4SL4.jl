conf_list = CSV.File("data/csv/EM_TB_ST4_SL4_Conf0.csv") |> Tables.matrix
conf_list_ = CSV.File("data/csv/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
conf_list = hcat(conf_list,conf_list_[:,[1:200...]])
St = 4; Sl = 4; pot = 4000;
X = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
X = reduce(hcat,X)
# ------------- Optimize a beta for each voltage ----------
β_list = []
for pot in [1000,2000,3000,4000,5000]
    k = 3
    conf_list = CSV.File("data/csv/EM_TB_ST4_SL4_Conf0.csv") |> Tables.matrix
    conf_list_ = CSV.File("data/csv/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
    conf_list = hcat(conf_list,conf_list_[:,[1:200...]])
    St = 4; Sl = 4;
    X = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
    X = reduce(hcat,X)
    β_min = optimize(Objective, 0.0, 1500.0, GoldenSection(),f_tol=1.0e-6,iterations=50)
    push!(β_list,β_min.minimizer)
end
println(β_list)
β_list = [816.077739050131, 46.97993387012737, 5.5027923359124, 9.563901355786785e-7, 4.515461700136166e-7]

# ----------------- kPCA ----------------------
# k = 3
# β = 47.09735445305706
# β = 0.030211707819700367
# β_min = optimize(Objective, 0.0, 1500.0, GoldenSection(),abs_tol=1.0e-8,iterations=25)
# β = β_min.minimizer
β = β_list[4]
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
plotlyjs()
gr()
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3")
# ----------------------- isomap ---------------------
neighbors = 25
Y_, D_G_sym = isomap1(neighbors,Z_)
s = scatter(eachrow(Y_)...,xlabel="y1",ylabel="y2",label="TS",legend=:outerbottom,legend_columns=2)
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
# Input test data ------------------------------------
conf_list_test = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
conf_list_test = conf_list_test[:,[1000-9:1000...]]

# ------------ BCRM for single shape and err for test shapes ----------------
nh = 1
n_test = 2
conf = conf_list_test[:,n_test]
x_test = ReadData_i(St,Sl,conf,pot)
y_gen = VectorSearch(Y_,conf,VS_Conf_list)
plot_x(x_test,"FOS TestID=$n_test conf=$conf")
x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,nh)
plot_x!(x_gen,"FOS generated Simple ReverseMap")
x_gen = ReverseMap2(y_gen,conf,Y_,X,conf_list,nh)
plot_x!(x_gen,"FOS generated BCRM")
xlims!(-maximum(vcat(x_test,x_gen)),maximum(vcat(x_test,x_gen)))
ylims!(-maximum(vcat(x_test,x_gen)),maximum(vcat(x_test,x_gen)))
zlims!(-maximum(vcat(x_test,x_gen)),maximum(vcat(x_test,x_gen)))
Err = norm(x_gen-x_test)/norm(x_test)
Err = norm(x_gen[[30:87...]]-x_test[[30:87...]])/norm(x_test[[30:87...]])
scatter(x_gen[[30:87...]],x_test[[30:87...]])

function test1(nh,n_test,pot)
    conf = conf_list_test[:,n_test]
    x_test = ReadData_i(St,Sl,conf,pot)
    y_gen = VectorSearch(Y_,conf,VS_Conf_list)
    plot_x(x_test,"FOS TestID=$n_test conf=$conf")
    x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,nh)
    plot_x!(x_gen,"FOS generated Simple ReverseMap")
    x_gen = ReverseMap2(y_gen,conf,Y_,X,conf_list,nh)
    plot_x!(x_gen,"FOS generated BCRM")
    xlims!(-maximum(vcat(x_test,x_gen)),maximum(vcat(x_test,x_gen)))
    ylims!(-maximum(vcat(x_test,x_gen)),maximum(vcat(x_test,x_gen)))
    zlims!(-maximum(vcat(x_test,x_gen)),maximum(vcat(x_test,x_gen)))
    Err = norm(x_gen-x_test)/norm(x_test)
    Err = norm(x_gen[[30:87...]]-x_test[[30:87...]])/norm(x_test[[30:87...]])
    scatter(x_gen[[30:87...]],x_test[[30:87...]])
end

test1(3,1,4000)

Err_1 = []
Err_2 = []
for nh in 1:1
    for n_test in 1:10
        x_test = ReadData_i(St,Sl,conf_list_test[:,n_test],pot)
        conf = conf_list_test[:,n_test] # rand(eachcol(conf_complete))
        y_gen = VectorSearch(Y_,conf,VS_Conf_list)
        x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,nh)
        push!(Err_1,norm(x_gen[[30:87...]]-x_test[[30:87...]])/norm(x_test[[30:87...]]))
        x_gen = ReverseMap2(y_gen,conf,Y_,X,conf_list,nh)
        push!(Err_2,norm(x_gen[[30:87...]]-x_test[[30:87...]])/norm(x_test[[30:87...]]))
    end
end
plot([Err_1,Err_2],type=:bar)
plot(Err_2,type=:bar)
Err_1 = []
Err_2 = []
for pot in [1000,2000,3000,4000,5000]
    X = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
    X = reduce(hcat,X)
    β = β_list[Int(pot/1000)]
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
    Z_ = real.(U_'*Ḡ)
    neighbors = 25
    Y_, D_G_sym = isomap1(neighbors,Z_)
    for nh in 1:1
        for n_test in 1:10
            x_test = ReadData_i(St,Sl,conf_list_test[:,n_test],pot)
            conf = conf_list_test[:,n_test] # rand(eachcol(conf_complete))
            y_gen = VectorSearch(Y_,conf,VS_Conf_list)
            x_gen, w_ns, Z_ns = ReverseMap(X,Y_,y_gen,nh)
            push!(Err_1,norm(x_gen[[30:87...]]-x_test[[30:87...]])/norm(x_test[[30:87...]]))
            x_gen = ReverseMap2(y_gen,conf,Y_,X,conf_list,nh)
            push!(Err_2,norm(x_gen[[30:87...]]-x_test[[30:87...]])/norm(x_test[[30:87...]]))
        end
    end
end

nh = 1
NH = []
E_ = []
p = [plot_x(ReadData_i(St,Sl,conf_list_test[:,n_test],pot),"FOS ID test = $n_test") for n_test in 1:10, pot in [1000,2000,3000,4000,5000]]
for pot in [1000,2000,3000,4000,5000]
    println("Potential currently running $pot")
    X = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
    X = reduce(hcat,X)
    β = β_list[Int(pot/1000)]
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
    Z_ = real.(U_'*Ḡ)
    neighbors = 25
    Y_, D_G_sym = isomap1(neighbors,Z_)
    Y_complete = []
    for i in 1:lastindex(eachcol(conf_complete))
        conf = conf_complete[:,i]
        push!(Y_complete,VectorSearch(Y_,conf,VS_Conf_list))
    end
    Y_complete = reduce(hcat,Y_complete)
    for n_test in 1:10
        println("Test config currently running $n_test")
        x_test = ReadData_i(St,Sl,conf_list_test[:,n_test],pot)
        # x_test = X_test[:,n_test]
        p[n_test,Int(pot/1000)] = plot_x(x_test,"FOS ID test = $n_test")
        y_new = NewData(x_test,Κ,neighbors,G,Z_,D_G_sym)
        # s = scatter!([y_new[1]],[y_new[2]],label="ID test = $n_test in RS")
        kdtree = KDTree(Y_complete; leafsize = 10)
        o = 500
        idxs, dists = knn(kdtree, y_new, o, true)
        Err_ = []
        for idx in idxs
            x_gen = ReverseMap2(Y_complete[:,idx],
            conf_complete[:,idx],Y_,X,conf_list,nh)
            # p = plot_x!(x_gen)
            push!(Err_,norm(x_gen[[30:87...]]-x_test[[30:87...]])/norm(x_test[[30:87...]]))
            # push!(Err_,Κ(x_test,x_gen))
            print("\ridx running = $idx")
        end
        # plts[n_test] = p
        Indices = sortperm(Err_)
        x_gen = ReverseMap2(Y_complete[:,idxs[Indices[1]]], conf_complete[:,idxs[Indices[1]]],Y_,X,conf_list,nh)
        p[n_test,Int(pot/1000)] = plot_x!(x_gen,"FOSgen of ID test = $n_test")
        # -----------------  FOS via simulation   --------------------------------------------------------------------
        # ph, chache = main(; get_parameters(2000.0, conf_complete[:,idxs[Indices[1]]], 4, 4)...)
        # x_line = CenterLine(ph)
        # x_line_list = [x_line]
        # x_line_list = [getproperty.(x_line,:data) for x_line in x_line_list]
        # x_line_list = [x_line_list[i][j][k] for i in 1:lastindex(x_line_list), j in 1:lastindex(x_line_list[1]), k in 1:lastindex(x_line_list[1][1])]
        # df = DataFrame(vcat(x_line_list[:,:,1],x_line_list[:,:,2],x_line_list[:,:,3]), :auto)
        # CSV.write("data/csv/EM_TB_St4_Sl4_test_$n_test.csv", df)
        # _X_test = CSV.File("data/csv/EM_TB_St4_Sl4_test_$n_test.csv") |> Tables.matrix
        # _X_test = _X_test'
        # n = 1
        # X_test_gen = []
        # push!(X_test_gen,_X_test[:,[1:n...]])
        # push!(X_test_gen,_X_test[:,[n+1:2*n...]])
        # push!(X_test_gen,_X_test[:,[2*n+1:3*n...]])
        # X_test_gen = reduce(vcat,X_test_gen)
        # p[n_test] = plot_x!(X_test_gen,"FOS of found PP for ID test = $n_test")
        # push!(E_,(norm(x_test-X_test_gen))/norm(x_test))
        # -----------------------------------------------------------------------------------------
        id_conf = 0
        id_idx = 0
        for i in 1:lastindex(Indices)
            if conf_complete[:,idxs[Indices[i]]]==conf_list_test[:,n_test]
                id_conf = i
                id_idx = idxs[Indices[i]]
                x_gen = ReverseMap2(Y_complete[:,idxs[Indices[i]]], conf_complete[:,idxs[Indices[i]]],Y_,X,conf_list,nh)
                p[n_test,Int(pot/1000)] = plot_x!(x_gen,"FOSgen of ID test = $n_test")
                println("\n$i")
            end
        end
        push!(NH,id_conf)
        println("\n------------------------")
    end
end
plot(NH,type=:bar)
plot(E_,type=:bar,xlabel="Test ID",ylabel="Error norm/norm",label=false)
display(p[n_test,Int(pot/1000)])