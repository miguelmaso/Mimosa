using LinearAlgebra
plotlyjs()
X = run()
k=3
β = 1.0e-3
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)

scatter(eachrow(Z_)...)
plot_eigen(Λ)

function kPCA_β_test(β)
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
    Z_ = real.(U_'*Ḡ)
    Λ = real.(Λ)
    Λ_t = sum(Λ)
    Λ = (1/Λ_t)*Λ
    return Λ[1]+Λ[2]+Λ[3]
end

β_list = collect(0.0:5.0e-4:1e-2)
Λ_sum_list = kPCA_β_test.(β_list)
bar(β_list,Λ_sum_list)

k=3
β = 0.0000025
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)

scatter(eachrow(Z_)...)
plot_eigen(Λ)



# G = X'*X
# l = eigen(G)
# f(λ)=-abs(λ)
# Λ_ = eigen(G,sortby=f)
# Λ = Λ_.values
# U = Λ_.vectors
# Σ = diagm(real.(sqrt.(complex(Λ))))
# Σ_ = Σ[:,[1:k...]]
# U_ = U*pinv(Σ_')
# Z_ = real.(U_'*Ḡ)

neighbors = 100
Y_, D_G_sym = isomap1(neighbors,Z_)
scatter(eachrow(Y_)...)

C = []
for i1 in 0:1, i2 in 0:1, i3 in 0:1, i4 in 0:1, i5 in 0:1, i6 in 0:1, i7 in 0:1, i8 in 0:1
    push!(C,[i1,i2,i3,i4,i5,i6,i7,i8])
end
C = reduce(hcat,C)

conf_a = [digits(i-1,base=2,pad=2) for i in 1:4]
conf_a = reduce(hcat,conf_a)
conf_VS = hcat(vcat(conf_a,zeros(6,4)),vcat(zeros(2,4),conf_a,zeros(4,4)),vcat(zeros(4,4),conf_a,zeros(2,4)),vcat(zeros(6,4),conf_a))

n,m = size(C)
VS_list = []
for c in eachcol(conf_VS)
    count = 1
    for cc in eachcol(C)
        if c==cc
            push!(VS_list,count)
            break            
        end
        count += 1
    end
end

Y_gen = []
for conf_i in eachcol(C)
    y_gen = VectorSearch(Y_,conf_i,VS_list)
    push!(Y_gen,y_gen)
end
Y_gen = reduce(hcat,Y_gen)
scatter!(eachrow(Y_gen)...)

error = norm(Y_gen-Y_)/norm(Y_gen)

function Obj(β)
    println("β parameter = $β")
    k=3
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
    Z_ = real.(U_'*Ḡ)
    neighbors = 100
    Y_, D_G_sym = isomap1(neighbors,Z_)
    Y_gen = []
    for conf_i in eachcol(C)
        y_gen = VectorSearch(Y_,conf_i,VS_list)
        push!(Y_gen,y_gen)
    end
    Y_gen = reduce(hcat,Y_gen)
    scatter!(eachrow(Y_gen)...)

    error = norm(Y_gen-Y_)/norm(Y_gen)
    println("Error = $error")
    return error
end

β_min = optimize(Obj, 0.0, 1.0e-4, GoldenSection(),f_tol=1.0e-6,iterations=20)

β_min.minimizer

β = 1.0696331036034351e-8

Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)

scatter(eachrow(Z_)...)
plot_eigen(Λ)

neighbors = 100
Y_, D_G_sym = isomap1(neighbors,Z_)
scatter(eachrow(Y_)...)

Y_gen = []
for conf_i in eachcol(C)
    y_gen = VectorSearch(Y_,conf_i,VS_list)
    push!(Y_gen,y_gen)
end
Y_gen = reduce(hcat,Y_gen)
scatter!(eachrow(Y_gen)...)

error = norm(Y_gen-Y_)/norm(Y_gen)

k = 2
Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)

scatter(eachrow(Z_)...)
plot_eigen(Λ)

Z_gen = []
for conf_i in eachcol(C)
    z_gen = VectorSearch(Z_,conf_i,VS_list)
    push!(Z_gen,z_gen)
end
Z_gen = reduce(hcat,Z_gen)
scatter!(eachrow(Z_gen)...)

error = norm(Z_gen-Z_)/norm(Z_gen)