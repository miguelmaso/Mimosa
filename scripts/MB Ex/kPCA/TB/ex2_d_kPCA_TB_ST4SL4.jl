conf_list = CSV.File("data/csv/EM_TB_ST4_SL4_Conf0.csv") |> Tables.matrix
conf_list_ = CSV.File("data/csv/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
conf_list = hcat(conf_list,conf_list_[:,[1:200...]])

St = 4; Sl = 4; pot = 2000;
X = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
X = reduce(hcat,X)
for pot in [3000,4000,5000]
    X_ = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
    X_ = reduce(hcat,X_)
    X = hcat(X,X_)
end
_, N = size(X)

plot_x(X[:,200],"$(conf_list[:,200]) @ 2000 V")
for i in 1:3
    plot_x!(X[:,200+264*i],"$(conf_list[:,200]) @ $((i+2)*1000) V")
end
zlims!(-0.1,0.1); ylims!(-0.1,0.1)

β_list = [816.077739050131, 46.97993387012737, 5.5027923359124, 9.563901355786785e-7, 4.515461700136166e-7]
β = sum([β_list[1],β_list[3],β_list[5]])/3
β = 100 # 5.5027923359124
β = β_list[5]
k=3
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
plot_eigen(Λ)
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3"
    ,color=[group_pot(i,N,4) for i in 1:N])
plotlyjs()
gr()

function group_pot(i,n,n_group)
    group_size = n/n_group
    for j in 1:n_group
        if (j-1)*group_size<i && i<=j*group_size
            return j
        end
    end
end

N = lastindex(eachcol(X))
neighbors = 25
Y_, D_G_sym = isomap1(neighbors,Z_)
s = scatter(eachrow(Y_)...,xlabel="y1",ylabel="y2",label="TS",legend=:outerbottom,
    legend_columns=2,
    color=[group_pot(i,N,4) for i in 1:N])


function Objective6_(x,x_,X_)
    θ = x[1]
    α = x[2]
    _x = x[3]
    _y = x[4]
    _, n = size(X_)
    rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
    scale = [α 0;0 α]
    trans = vcat(_x*ones(1,n),_y*ones(1,n))
    x_ = rot*x_
    x_ = scale*x_
    x_ = trans+x_
    println("rot $θ  scale $α x $_x y $_y")
    println("Diff $(norm(X_-x_)/norm(X_))")
    return norm(X_-x_)/norm(X_)
end

Objective6(x) = Objective6_(x,Y_3000,Y_5000)
x0 = [0.0,1.0,0.0,0.0]
min = []
for j in 1:3
    Objective6(x) = Objective6_(x,Y_[:,[1+(j-1)*Int(N/4):j*Int(N/4)...]],Y_[:,[1+(4-1)*Int(N/4):4*Int(N/4)...]])
    _min = optimize(Objective6, x0, NelderMead())
    push!(min,_min)
end

θ_list = [min[i].minimizer[1] for i in 1:3]
α_list = [min[i].minimizer[2] for i in 1:3]



_min = optimize(Objective6, x0, NelderMead())
θ = _min.minimizer[1]
α = _min.minimizer[2]
_x = _min.minimizer[3]
_y = _min.minimizer[4]
rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
scale = [α 0;0 α]
_, n = size(Y_3000)
trans = vcat(_x*ones(1,n),_y*ones(1,n))
Y_gen = rot*Y_3000
Y_gen = scale*Y_gen
Y_gen = trans+Y_gen
s = scatter(eachrow(Y_5000)...,xlabel="y1",ylabel="y2")
s = scatter!(eachrow(Y_gen)...,xlabel="y1",ylabel="y2")



Objective6(x) = Objective6_(x,Y_1000,Y_5000)
x0 = [0.0,1.0]
_min = optimize(Objective6, x0, NelderMead())
θ = _min.minimizer[1]
α = _min.minimizer[2]
rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
scale = [α 0;0 α]
Y_gen = rot*Y_1000
Y_gen = scale*Y_gen
_min = optimize(Objective6, _min.minimizer, NelderMead())
s = scatter(eachrow(Y_5000)...,xlabel="y1",ylabel="y2")
s = scatter!(eachrow(Y_gen)...,xlabel="y1",ylabel="y2")

VS_Conf_list = [1:64...]
Y_gen = []
for j in 1:4
    for i in 1:lastindex(eachcol(conf_list))
        conf = conf_list[:,i]
        push!(Y_gen,VectorSearch(Y_[:,[1+(j-1)*Int(N/4):j*Int(N/4)...]],conf,VS_Conf_list))
    end
end
Y_gen = reduce(hcat,Y_gen)
scatter!(eachrow(Y_gen)...,xlabel="y1",ylabel="y2",label="TS gen")
Err_TS = norm(Y_gen-Y_)/norm(Y_)

conf_complete = []
for i1 in 0:1, i2 in 0:1, i3 in 0:1, i4 in 0:1, i5 in 0:1, i6 in 0:1, i7 in 0:1, i8 in 0:1, i9 in 0:1, i10 in 0:1, i11 in 0:1, i12 in 0:1, i13 in 0:1, i14 in 0:1, i15 in 0:1, i16 in 0:1
    push!(conf_complete,[i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16])
end
conf_complete = reduce(hcat,conf_complete)
Y_complete = []
for j in 1:1
    for i in 1:lastindex(eachcol(conf_complete))
        conf = conf_complete[:,i]
        push!(Y_complete,VectorSearch(Y_[:,[1+(j-1)*Int(N/4):j*Int(N/4)...]],conf,VS_Conf_list))
    end
end
Y_complete = reduce(hcat,Y_complete)
scatter!(eachrow(Y_complete)...,xlabel="y1",ylabel="y2")