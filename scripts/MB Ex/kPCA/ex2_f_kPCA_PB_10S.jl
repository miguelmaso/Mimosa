RTS_parts,STS_parts,Phi = 9, 0, 2000
RTS_parts,STS_parts,RTS_colums,STS_columns,Phi = 9, 15, 922, 0, 2000
X, conf = ReadData_RTS_STS(RTS_parts,STS_parts,Phi)
X, conf = ReadData_RTS_STS(RTS_parts,STS_parts,RTS_colums,STS_columns,Phi)
n = lastindex(eachcol(X))
# Smart TS 200 optimization
β = 0.15806570088783478 
# Kernel Function
Κ(X1,X2) = (X1'*X2 + β)^2
# kPCA
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
# isomap
neighbors = 25
Y_, D_G_sym = isomap1(neighbors,Z_)
# VS generation of TS for comparison
scatter(eachrow(Y_)...,xlabel="y1",ylabel="y2",label="TS")
Y_gen = []
for i in 1:n
    y_ = VectorSearch(Y_,conf[:,i])
    push!(Y_gen,y_)
end
Y_gen = reduce(hcat,Y_gen)
Err_TS_1 = norm(Y_gen-Y_)/norm(Y_)
scatter!(eachrow(Y_gen)...,xlabel="y1",ylabel="y2",label="TS")


n_ = 6+1
Err_TS = zeros(n_,n_)
for i in 5:n_, j in 1:n_
    println("i = $i")
    RTS_parts,STS_parts,RTS_colums,STS_columns,Phi = 12, 15, 622+(i-1)*100, (j-1)*100, 2000
    X, conf = ReadData_RTS_STS(RTS_parts,STS_parts,RTS_colums,STS_columns,Phi)
    n = lastindex(eachcol(X))
    # Smart TS 200 optimization
    β = 0.15806570088783478 
    # Kernel Function
    Κ(X1,X2) = (X1'*X2 + β)^2
    # kPCA
    k = 3
    Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
    Z_ = real.(U_'*Ḡ)
    # isomap
    neighbors = 25
    Y_, D_G_sym = isomap1(neighbors,Z_)
    # VS generation of TS for comparison
    scatter(eachrow(Y_)...,xlabel="y1",ylabel="y2",label="TS")
    Y_gen = []
    for i in 1:n
        y_ = VectorSearch(Y_,conf[:,i])
        push!(Y_gen,y_)
    end
    Y_gen = reduce(hcat,Y_gen)
    Err_TS[i,j] = norm(Y_gen-Y_)/norm(Y_)
    scatter!(eachrow(Y_gen)...,xlabel="y1",ylabel="y2",label="TS")
end
heatmap([(j-1)*100 for j in 1:n_],[622+(i-1)*100 for i in 1:n_],Err_TS,
xticks= [(j-1)*100 for j in 1:n_], yticks = [622+(i-1)*100 for i in 1:n_], c=cgrad([:blue, :red]),xlabel="Smart TS increase",ylabel="Random TS increase")
df = DataFrame(Err_TS, :auto)
CSV.write("data/csv/Err_TSgen_RandomlyVSSmartlyTSIncrease.csv", df)
p = plot([(j-1)*100 for j in 1:n_],Err_TS[1,:],label="TS generation err increaseing TS using Smartly selected conf", xlabel="increase in the TS",ylabel="Error norm/norm of regenerating the TS")
p = plot([(j-1)*100 for j in 1:n_],Err_TS[:,1],label="TS generation err increaseing TS using Randonly selected conf")
p = plot([622+(i-1)*100 for i in 1:n_],Err_TS[:,1])
jj = [(i-1)*100 for i in 1:n_]
for i in 2:7
    P = plot!([(j-1)*100 for j in 1:n_],Err_TS[:,i],label="starting from $(jj[i]) Smartly selected points")
end
display(p)