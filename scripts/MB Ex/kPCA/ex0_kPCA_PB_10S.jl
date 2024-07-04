using CSV
using Tables
using Statistics
using LinearAlgebra
using Plots
plotlyjs()

# Phi = [2000,3000,4000]
function ReadData(n_parts,Phi,d)
    X = []
    for i in 1:n_parts, j in Phi
        push!(X,CSV.File("data/csv/EM_PB_10S_Phi$j/EM_PB_10S_$i.csv") |> Tables.matrix)
    end
    X = hcat(X...)
    if d
        m, n = size(X)
        z = zeros(1,n)
        X1 = vcat(X,z)
        X2 = vcat(z,X)
        X3 = (1/0.001)*(X1-X2)
        X = X3[[2:m...],:]
    end
    return X
end


function PCA(X,k)
    m, n = size(X)
    X̄ = [1. for i in 1:n]
    mean!(X̄',X)
    for i in 1:n
        X[:,i] = X[:,i].-X̄[i]
    end
    C = X*X'
    f(λ)=-λ
    Λ_ = eigen(C,sortby=f)
    Λ = Λ_.values
    U = Λ_.vectors
    U_ = U[:,1:k]
    return Λ, U, U_
end

function POD(X,k)
    m, n = size(X)
    X̄ = [1. for i in 1:n]
    mean!(X̄',X)
    for i in 1:n
        X[:,i] = X[:,i].-X̄[i]
    end
    G = X'*X
    f(λ)=-λ
    Λ_ = eigen(G,sortby=f)
    Λ = Λ_.values
    U = Λ_.vectors
    U_ = U[:,1:k]
    return Λ, U, U_
end

function kPOD(Κ,X,k)
    m, n = size(X)
    X̄ = [1. for i in 1:n]
    mean!(X̄',X)
    for i in 1:n
        X[:,i] = X[:,i].-X̄[i]
    end
    G = [Κ(X[:,i],X[:,j]) for i in 1:n, j in 1:n]
    # display(G)
    I = ones(n,n)
    Ḡ = G - (1/n)*G*I - (1/n)*I*G + (1/n^2)*I*G*I 
    # display(Ḡ)
    f(λ)=-abs(λ)
    Λ_ = eigen(Ḡ,sortby=f)
    Λ = Λ_.values
    U = Λ_.vectors
    U_ = U[:,1:k]
    return Λ, U, U_, Ḡ
end

function PlotSetupN(X_,k)
    X_pl = []
    for i in 1:k
        push!(X_pl,X_[i,:])
    end
    N_pl = [i for i in 1:length(X_pl[1])]
    pushfirst!(X_pl,N_pl)
    return X_pl
end

function PlotSetup(X_,k)
    X_pl = []
    for i in 1:k
        push!(X_pl,X_[i,:])
    end
    return X_pl
end

function plotPCA(n_parts,Phi, U_,k,d)
    l = length(Phi)
    X = ReadData(n_parts,Phi,d)
    m, n = size(X)
    l = n/l
    Z_ = U_'*X
    Z_pl = PlotSetup(Z_,k)
    s = scatter(Z_pl...,xlabel="z1",ylabel="z2",zlabel="z3", markercolor= [Int(ceil(i/l)) for i in 1:n])
    display(s)
end

function plotPOD(n_parts,Phi,U_,k,d)
    l = length(Phi)
    X = ReadData(n_parts,Phi,d)
    m, n = size(X)
    l = n/l
    Z_ = U_'*(X'*X) #Corregir uso de G y usar la ecuación 2.28 del capitulo del libro
    # Z_ = abs.(Z_)
    Z_ = real.(Z_)
    Z_pl = PlotSetup(Z_,k)
    s = scatter(Z_pl...,xlabel="z1",ylabel="z2",zlabel="z3", hover= [i for i in 1:n])#, markercolor= [Int(ceil(i/l)) for i in 1:n])
    display(s)
end

function plotkPOD(Ḡ,U_,k,d)
    m, n = size(Ḡ)
    Z_ = U_'*Ḡ #Corregir uso de G y usar la ecuación 2.28 del capitulo del libro
    # Z_ = abs.(Z_)
    Z_ = real.(Z_)
    Z_pl = PlotSetup(Z_,k)
    s = scatter(Z_pl...,xlabel="z1",ylabel="z2",zlabel="z3", hover= [i for i in 1:n])#, markercolor= [Int(ceil(i/l)) for i in 1:n])
    display(s)
end

function execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Λ_t = sum(Λ)
    Λ = (1/Λ_t)*Λ
    Λ_s = round.(100*Λ[[i for i in 1:10]])
    p = plot(real.(Λ_s),type="bar",xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
    display(p)
    if k<=3
        plotkPOD(Ḡ,U_,k,d)
    else
        plotkPOD(Ḡ,U_,3,d)
    end
    return X, U_, Ḡ
end

function execute_POD(P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_ = POD(X,k)
    Λ_t = sum(Λ)
    Λ = (1/Λ_t)*Λ
    Λ_s = round.(100*Λ[[i for i in 1:10]])
    p = plot(Λ_s,type="bar",xlabel="λ",ylabel="%", legend=false, hover=Λ_s)
    display(p)
    if k<=3
        plotPOD(n_parts,P,U_,k,d)
    else
        plotPOD(n_parts,P,U_,3,d)
    end
end

function execute_PCA(P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_ = PCA(X,k)
    Λ_t = sum(Λ)
    Λ = (1/Λ_t)*Λ
    Λ_s = round.(100*Λ[[i for i in 1:10]])
    p = plot(Λ_s,type="bar",xlabel="λ",ylabel="%", legend=false, hover=Λ_s)
    display(p)
    if k<=3
        plotPCA(n_parts,P,U_,k,d)
    else
        plotPCA(n_parts,P,U_,3,d)
    end
    return X, U_
end

function ReverseMap(U_,X,z,d)
    Z_ = U_'*(X'*X)
    m, n = size(U_)
    w = [1/sqrt(dot(z-zi,z-zi)) for zi in eachcol(Z_)]
    # w = [exp(-sqrt(dot(z-zi,z-zi))) for zi in eachcol(Z_)]
    # display(w)
    w_sort = sortperm(w, rev=true)
    w_ns = [w[w_sort[1]]]
    X_ns = X[:,w_sort[1]]
    Z_ns = Z_[:,w_sort[1]]
    for i in 2:8
        push!(w_ns,w[w_sort[i]])
        X_ns = hcat(X_ns,X[:,w_sort[i]])
        Z_ns = hcat(Z_ns,Z_[:,w_sort[i]])
    end
    display(X_ns)
    display(w_ns)
    # w_ns = w_ns/sum(w_ns)
    w_ns = pinv(Z_ns)*z
    w_ns = w_ns/sum(w_ns)
    display(w_ns)
    x = X_ns*w_ns
    if d
        y = [0.004]
        n = length(x)
        for i in 1:n
            push!(y,y[i]+0.001*real.(x[i]))
        end
        x=y
    end
    return x, w_ns, Z_ns
end

function VectorSearch(U_,X,conf)
    l = length(conf)
    Z_ = U_'*(X'*X)
    z = Z_[:,[end-l:end...]]
    v = [z[:,1]-z[:,i] for i in 2:l+1]
    z = z[:,1]
    for i in 1:length(conf)
        if conf[i]==0
            conf[i] = 0
        elseif conf[i]==2
            conf[i] = 1
        elseif conf[i]==1
            conf[i] = -1
        end
    end
    for i in 1:length(conf)
        z = z + conf[i]*v[end+1-i]
    end
    return z
end

function Distance0(β)
    P = [4000] #,4000]
    n_parts = 7
    k = 3
    d = false
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    X, U_ = execute_kPOD(Κ,P,n_parts,k,d)
    Z_ = U_'*(X'*X)
    dist = sqrt(dot(real.(Z_[:,601]-Z_[:,602]),real.(Z_[:,601]-Z_[:,602])))
    return 1/dist
end

P = [4000] #,4000]
n_parts = 7
k = 3
d = false
#=
#β = 4.146016e-02
β = 0.9
Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2))); X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d);
Κ(X1,X2) = exp(-0.8*(dot(X1-X2,X1-X2))); X, U_ = execute_kPOD(Κ,P,n_parts,k,d);
Κ(X1,X2) = (X1'*X2 + 41.95)^2; X, U_ = execute_kPOD(Κ,P,n_parts,k,d);
Κ(X1,X2) = (X1'*X2 + 3.1)^4; X, U_ = execute_kPOD(Κ,P,n_parts,k,d);

X, U_ = execute_PCA(P,n_parts,k,d)
execute_POD(P,n_parts,k,d)
X, U_ = execute_kPOD(Κ,P,n_parts,k,d);
Z_ = U_'*(X'*X)
z = -Z_[:,38]
x, w_ns, Z_ns = ReverseMap(real.(U_),X,z,d)
plot(x)
plot(X[:,1])
scatter!([z[1]],[z[2]],[z[3]])
z = VectorSearch(U_,X,conf)
z = real.(z)
conf = [0,0,0,0,0,0,0,0,1,1]
conf = [0,0,0,0,0,2,0,2,2,2]; z = VectorSearch(U_,X,conf); z = real.(z); scatter!([z[1]],[z[2]])#,[z[3]])
error = sqrt(dot(X[:,1]-x,X[:,1]-x))
error = sqrt(dot(X_11-x,X_11-x))
=#
#=
using Optim
β_min = optimize(Distance0, 1e-10, 1e1, GoldenSection())
=#