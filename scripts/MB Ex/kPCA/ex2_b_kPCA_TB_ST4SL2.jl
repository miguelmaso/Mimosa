j = 2000
i = 3
X = CSV.File("data/csv/EM_TB_ST4SL2_Phi$j/EM_TB_St4_Sl2_$i.csv") |> Tables.matrix
_X = _X'
n = 64
X_ = []
push!(X_,_X[:,[1:n...]])
push!(X_,_X[:,[n+1:2*n...]])
push!(X_,_X[:,[2*n+1:3*n...]])
X_[1] = hcat(X_[1],_X[:,[1:n...]])
X_[2] = hcat(X_[2],_X[:,[n+1:2*n...]])
X_[3] = hcat(X_[3],_X[:,[2*n+1:3*n...]])
conf = 4
plot(X_[1][:,conf],X_[2][:,conf],X_[3][:,conf])
X = reduce(vcat,X_)
k = 2
β = 10
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Κ(X1,X2) = (X1'*X2 + β)^2
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
scatter(eachrow(Z_)..., hover = [i for i in 1:lastindex(eachcol(Z_))],
xlabel="z1",ylabel="z2",zlabel="z3")
Λ = real.(Λ)
Λ_t = sum(Λ)
Λ = (1/Λ_t)*Λ
Λ_s = round.(100*Λ[[i for i in 1:10]])
p = plot(real.(Λ_s),type="bar",xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
display(p)