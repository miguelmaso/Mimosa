using CSV
using Tables
using Statistics
using LinearAlgebra
using Plots
plotlyjs()

# Phi = [2000,3000,4000]
function ReadData(n_parts,Phi)
    X = []
    for i in 1:n_parts, j in Phi
        push!(X,CSV.File("data/csv/EM_PB_4S_Phi$j/EM_PB_4S_$i.csv") |> Tables.matrix)
    end
    X = hcat(X...)
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
    return Λ, U, U_
end

function PlotSetupN(X_,k)
    X_pl = []
    for i in 1:k
        push!(X_pl,X_[i,:])
    end
    N_pl = [i for i in 1:length(X_pl[1])]
    push!(X_pl,N_pl)
    return X_pl
end

function PlotSetup(X_,k)
    X_pl = []
    for i in 1:k
        push!(X_pl,X_[i,:])
    end
    return X_pl
end

function plotPCA(n_parts,Phi, U_,k)
    l = length(Phi)
    X = ReadData(n_parts,Phi)
    m, n = size(X)
    Z_ = U_'*X
    Z_pl = PlotSetup(Z_,k)
    s = scatter(Z_pl...,xlabel="x1",ylabel="x2",zlabel="x3", markercolor= [Int(ceil(i/l)) for i in 1:n])
    display(s)
end

function plotPOD(n_parts,Phi,U_,k)
    l = length(Phi)
    X = ReadData(n_parts,Phi)
    m, n = size(X)
    Z_ = U_'*(X'*X)
    Z_pl = PlotSetup(Z_,k)
    s = scatter(Z_pl...,xlabel="x1",ylabel="x2",zlabel="x3", markercolor= [Int(ceil(i/l)) for i in 1:n])
    display(s)
end

function execute_kPOD(Κ,P,n_parts,k)
    X = ReadData(n_parts,P)
    Λ, U, U_ = kPOD(Κ,X,k)
    Λ_t = sum(Λ)
    Λ = (1/Λ_t)*Λ
    Λ_s = round.(100*Λ[[i for i in 1:10]])
    p = plot(real.(Λ_s),type="bar",xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
    display(p)
    plotPOD(n_parts,P,real.(U_),k)
end

function execute_POD(P,n_parts,k)
    X = ReadData(n_parts,P)
    Λ, U, U_ = POD(X,k)
    Λ_t = sum(Λ)
    Λ = (1/Λ_t)*Λ
    Λ_s = round.(100*Λ[[i for i in 1:10]])
    p = plot(Λ_s,type="bar",xlabel="λ",ylabel="%", legend=false, hover=Λ_s)
    display(p)
    plotPOD(n_parts,P,U_,k)
end

function execute_PCA(P,n_parts,k)
    X = ReadData(n_parts,P)
    Λ, U, U_ = PCA(X,k)
    Λ_t = sum(Λ)
    Λ = (1/Λ_t)*Λ
    Λ_s = round.(100*Λ[[i for i in 1:10]])
    p = plot(Λ_s,type="bar",xlabel="λ",ylabel="%", legend=false, hover=Λ_s)
    display(p)
    plotPCA(n_parts,P,U_,k)
end

Κ(X1,X2) = exp(-5*(dot(X1-X2,X1-X2)))
P = [2000,3000,4000]
n_parts = 3
k = 3
# execute_PCA(P,n_parts,k)
# execute_POD(P,n_parts,k)
# execute_kPOD(Κ,P,n_parts,k)