conf_list = CSV.File("data/csv/EM_TB_ST4_SL4_Conf0.csv") |> Tables.matrix
conf_list_ = CSV.File("data/csv/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
conf_list = hcat(conf_list,conf_list_[:,[1:200...]])
St = 4; Sl = 4; pot = 1000;
X = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
X = reduce(hcat,X)
pot = 3000
X_ = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
X_ = reduce(hcat,X_)
X = hcat(X,X_)
pot = 5000
X_ = [ReadData_i(St,Sl,conf,pot) for conf in eachcol(conf_list)]
X_ = reduce(hcat,X_)
X = hcat(X,X_)

β_list = [816.077739050131, 46.97993387012737, 5.5027923359124, 9.563901355786785e-7, 4.515461700136166e-7]
β = sum([β_list[1],β_list[3],β_list[5]])/3
β = 100 # 5.5027923359124
k=6
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
plot_eigen(Λ)
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3"
    ,color=[group_pot(i,N,3) for i in 1:N])
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
    color=[group_pot(i,N,3) for i in 1:N])
