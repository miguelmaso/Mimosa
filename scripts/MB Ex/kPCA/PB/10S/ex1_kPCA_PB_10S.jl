using CSV
using Tables
using Statistics
using LinearAlgebra
using Plots
using Optim
plotlyjs()

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

function kPOD(Κ,X,k)
    m, n = size(X)
    # X̄ = [1. for i in 1:n]
    # mean!(X̄',X)
    # for i in 1:n
    #     X[:,i] = X[:,i].-X̄[i]
    # end
    G = [Κ(X[:,i],X[:,j]) for i in 1:n, j in 1:n]
    # display(G)
    I = ones(n,n)
    Ḡ = G - (1/n)*G*I - (1/n)*I*G + (1/n^2)*I*G*I 
    # display(Ḡ)
    f(λ)=-abs(λ)
    Λ_ = eigen(Ḡ,sortby=f)
    Λ = Λ_.values
    U = Λ_.vectors
    Σ = diagm(real.(sqrt.(complex(Λ))))
    Σ_ = Σ[:,[1:k...]]
    U_ = U*pinv(Σ_')
    # U_ = U[:,1:k]
    return Λ, U, U_, Ḡ
end

function PlotSetup(X_,k)
    X_pl = []
    for i in 1:k
        push!(X_pl,X_[i,:])
    end
    return X_pl
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

function ReverseMap(Z_,z,d,n)
    w = [1/sqrt(dot(z-zi,z-zi)) for zi in eachcol(Z_)]
    # w = [exp(-sqrt(dot(z-zi,z-zi))) for zi in eachcol(Z_)]
    # display(w)
    w_sort = sortperm(w, rev=true)
    w_ns = [w[w_sort[1]]]
    X_ns = X[:,w_sort[1]]
    Z_ns = Z_[:,w_sort[1]]
    for i in 2:n
        push!(w_ns,w[w_sort[i]])
        X_ns = hcat(X_ns,X[:,w_sort[i]])
        Z_ns = hcat(Z_ns,Z_[:,w_sort[i]])
    end
    # w_ns = w_ns/sum(w_ns)
    w_ns = pinv(Z_ns)*z
    w_ns = w_ns/sum(w_ns)
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

function RotM(v1,v2)
    v = cross(v1,v2)
    vₓ = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
    c = dot(v1,v2)
    R = I + vₓ + (1/(1+c))*vₓ*vₓ
    return R
end

function VectorSearch(Z_,conf)
    z = Z_[:,[end-l:end...]]
    v = [z[:,1]-z[:,i] for i in 2:l+1]
    z = z[:,1]
    # println(v)
    z = z0
    # println(z0)
    R = I
    u_0 = ∇zᵤ(z[1],z[2])
    for i in 1:lastindex(conf)
        if conf[i]==2
            conf[i] = 0
        elseif conf[i]==0
            conf[i] = -1
        elseif conf[i]==1
            conf[i] = 1
        end
    end
    # println(conf)
    for i in [1,2,3,4]
        # v_ = v[i]
        # v_[1] = conf[i]*v_[1]
        if conf[i] == 0
            z = z
        elseif conf[i]==1
            z = z + R*v[i]
        elseif conf[i]==-1
            z = z + R*v[4+i]
        end
    end
    return z
end

function VectorSearch(Z_,conf,∇zᵤ)
    z = Z_[:,[end-21:end...]]
    v = [z[:,i]-z[:,1] for i in 2:21]
    z = z[:,1]
    R = I
    u_0 = ∇zᵤ(z[1],z[2])
    for i in 1:lastindex(conf)
        # v_ = v[i]
        # v_[1] = conf[i]*v_[1]
        if conf[i] == 0
            z = z
        elseif conf[i]==1
            z = z + R*v[lastindex(conf)-i+1]
        elseif conf[i]==2
            z = z + R*v[2*lastindex(conf)-i+1]
        end
        # println(R)
        u_1 = ∇zᵤ(z[1],z[2])
        R = RotM(u_1,u_0)
        # u_0 = u_1
    end
    return z
end

function Objective1(β)
    println("Kernel parameter: $β")
    P = [4000] #,4000]
    n_parts = 8
    k = 3
    d = false
    # Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Κ(X1,X2) = (X1'*X2 + β)^2
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)
    final = lastindex(eachcol(Z_))
    D = Dif(Z_,final-21,final-12,final-11,final)
    return D
    # return 1-(abs(maximum(Z_[2,:]))/abs(maximum(Z_[1,:])))
end

function Objective2(β)
    println("Kernel parameter: $β")
    P = [2000] #,4000]
    n_parts = 6
    k = 3
    d = false
    # Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    Κ(X1,X2) = (X1'*X2 + β)^2
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    R = []
    err = []
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)

    # -------Curve fitting------
    # Quadric surface explicit in z3 of the form z3 = A + B*z1^2 + Cz2^2 + D*z1 + E*z2
    Quadric(z) = [1 z[1]^2 z[2]^2 z[1] z[2]] 

    # least squares fit of the curve to the TS in the RS
    A = [Quadric(Z) for Z in eachcol(Z_)]
    A = reduce(vcat,A)
    c = pinv(A)*Z_[3,:]
    
    #Definition of surface and unit gradient
    z3(z1,z2) = c[1] + c[2]*z1^2 + c[3]*z2^2 + c[4]*z1 + c[5]*z2
    ∇zᵤ(z1,z2) = (1/norm([2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]))*[2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]

    SS_reg = sqrt(dot(Z_[3,:]-z3.(Z_[1,:],Z_[2,:]),Z_[3,:]-z3.(Z_[1,:],Z_[2,:])))
    avr = sum(Z_[3,:])/length(Z_[3,:])
    SS_tot = sqrt(dot(Z_[3,:].-avr,Z_[3,:].-avr))
    R_sq = 1-SS_reg/SS_tot
    append!(R,sqrt(R_sq))
    println("R1 = $R")

    # Whole RS found using VectorSearch and considering ∇zᵤ
    z_ = []
    for c in eachcol(conf)
        z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
        push!(z_,z)
    end
    z_ = reduce(hcat,z_)

    # evaluation of the distributted error - Applicable if using the whole RS as the TS
    append!(err,norm(Z_-z_)/norm(Z_))
    println("Err1 = $err")

    Z_c = Z_[3,:]
    Z_[3,:] = Z_[2,:]
    Z_[2,:] = Z_c
    
    A = [Quadric(Z) for Z in eachcol(Z_)]
    A = reduce(vcat,A)
    c = pinv(A)*Z_[3,:]
    
    #Definition of surface and unit gradient
    z3(z1,z2) = c[1] + c[2]*z1^2 + c[3]*z2^2 + c[4]*z1 + c[5]*z2
    ∇zᵤ(z1,z2) = (1/norm([2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]))*[2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]

    SS_reg = sqrt(dot(Z_[3,:]-z3.(Z_[1,:],Z_[2,:]),Z_[3,:]-z3.(Z_[1,:],Z_[2,:])))
    avr = sum(Z_[3,:])/length(Z_[3,:])
    SS_tot = sqrt(dot(Z_[3,:].-avr,Z_[3,:].-avr))
    R_sq = 1-SS_reg/SS_tot
    append!(R,sqrt(R_sq))
    println("R2 = $R")

    # Whole RS found using VectorSearch and considering ∇zᵤ
    z_ = []
    for c in eachcol(conf)
        z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
        push!(z_,z)
    end
    z_ = reduce(hcat,z_)

    # evaluation of the distributted error - Applicable if using the whole RS as the TS
    append!(err,norm(Z_-z_)/norm(Z_))
    println("Err2 = $err")

    if R[1]>R[2]
        err = err[1]
        println("Err1 & R1")
    else
        err = err[2]
        println("Err2 & R2")
    end
    
    return err
end

Dist(Z_,i1,i2) = sqrt(dot(real.(Z_[:,i1]-Z_[:,i2]),real.(Z_[:,i1]-Z_[:,i2])))
Dif(Z_,i,j,k,l) = abs((Dist(Z_,k,l)-Dist(Z_,i,j))/Dist(Z_,i,j))

#=
Dif(Z_,601,610,611,622)

n_p, P, d = 6, [2000], false
# Training Set in the full-order space
X = ReadData(n_p, P, d)
# Number of principal directions to be considered
k = 3
#Gaussian Kernel
Κ(X1,X2) = exp(-3.0*(dot(X1-X2,X1-X2)))
# Polynomial Kernel
# Κ(X1,X2) = (X1'*X2 + 10)^2
Λ, U, U_, Ḡ = kPOD(Κ, X, k)

# Training Set in the reduced space
Z_ = real.(U_'*Ḡ)

#Generation of full set of parameters in order
conf = CSV.File("data/csv/EM_PB_10S_Phi2000/Config_N622_EM_PB_10S.csv") |> Tables.matrix

# Scatter plot of the TS in RS - Colored parameters
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))], markercolor = [c[1] for c in conf])

# Scatter plot of the TS in RS
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))], 
title="n_p, P = 622, [2000] <br> Κ(X1,X2) = exp(-3.0*(dot(X1-X2,X1-X2)))")
savefig("data/Figs/kPCA_PB_S10_Phi2000/RS_0.png")

# Plot normalized aigen values
Λ_t = sum(Λ)
Λ = (1/Λ_t)*Λ
Λ_s = round.(100*Λ[[i for i in 1:10]])
p = plot(real.(Λ_s),type="bar",xlabel="λ",ylabel="%", legend=false, hover=real.(Λ_s))
display(p)

# ------ Optimization to set a kernel parameter ------
β_min = optimize(Objective2, 0.00, 0.5, GoldenSection())
β = β_min.minimizer
# Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
Κ(X1,X2) = (X1'*X2 + β)^2
Λ, U, U_, Ḡ = kPOD(Κ, X, k)
Z_ = real.(U_'*Ḡ)
Dif(Z_,601,610,611,622)
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
    hover= [i for i in 1:lastindex(eachcol(Z_))], 
    title=" Κ(X1,X2) = (X1'*X2 + β)^2<br>β = $β")
savefig("data/Figs/kPCA_PB_S10_Phi2000/RS_Polynomial_Optimized.png")
scatter(eachrow(Z_[:,[601,610,611,622]])...)
DDD = []
space = range(start=0.0, step=1e-2, stop=1.0)
for b in space
    push!(DDD,Objective2(b))
end
plot(space,DDD, 
title="err,norm(Z_-z_)/norm(Z_) vs β <br> Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))")
savefig("data/Figs/kPCA_PB_S10_Phi2000/ObjVsBeta_Gaussiano.png")

# -------Curve fitting------
# Quadric surface explicit in z3 of the form z3 = A + B*z1^2 + Cz2^2 + D*z1 + E*z2
Quadric(z) = [1 z[1]^2 z[2]^2 z[1] z[2]] 

# Necessary when data fits to a surface explicit in z2 instead of z3
Z_c = Z_[3,:]
Z_[3,:] = Z_[2,:]
Z_[2,:] = Z_c

# least squares fit of the curve to the TS in the RS
A = [Quadric(Z) for Z in eachcol(Z_)]
A = reduce(vcat,A)
c = pinv(A)*Z_[3,:]

#Definition of surface and unit gradient
z3(z1,z2) = c[1] + c[2]*z1^2 + c[3]*z2^2 + c[4]*z1 + c[5]*z2
∇zᵤ(z1,z2) = (1/norm([2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]))*[2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]

# Plot of data and surface
scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))])
z1 = range(minimum(Z_[1,:]), stop=maximum(Z_[1,:]), length=100)
z2 = range(minimum(Z_[2,:]), stop=maximum(Z_[2,:]), length=100)
surface!(z1,z2,z3)
savefig("data/Figs/kPCA_PB_S10_Phi2000/RS_Polynomial_Optimized_SurfFit.png")

# Compute R for the fitted curve
SS_reg = sqrt(dot(Z_[3,:]-z3.(Z_[1,:],Z_[2,:]),Z_[3,:]-z3.(Z_[1,:],Z_[2,:])))
avr = sum(Z_[3,:])/length(Z_[3,:])
SS_tot = sqrt(dot(Z_[3,:].-avr,Z_[3,:].-avr))
R_sq = 1-SS_reg/SS_tot
R = sqrt(R_sq)

# Whole RS found using VectorSearch and considering ∇zᵤ
z_ = []
for c in eachcol(conf)
    z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
    push!(z_,z)
end
z_ = reduce(hcat,z_)
scatter!(eachrow(z_)..., hover= [i for i in 1:lastindex(eachcol(Z_))])
savefig("data/Figs/kPCA_PB_S10_Phi2000/RS_Polynomial_Optimized_SurfFit_Zgen.png")

# evaluation of the distributted error - Applicable if using the whole RS as the TS
norm(Z_-z_)/norm(Z_)

scatter(eachrow(Z_[:,[600:605...]])..., hover= [i for i in 600:lastindex(eachcol(Z_))])
scatter!(eachrow(z_[:,[600:605...]])..., hover= [i for i in 600:lastindex(eachcol(Z_))])

#ReverseMap calculation of configurations not in the TS
x, w_ns, Z_ns = ReverseMap(Z_,z_[:,6],d,8)
plot(x)
plot!(X[:,6])
savefig("data/Figs/kPCA_PB_S10_Phi2000/ReverseMap_X6inTS_8Vecino.png")
conf[:,6]
norm(x-X[:,6])/norm(X[:,6])

e = []
for j in 1:100
    err = []
    for i in 1:600
        x, w_ns, Z_ns = ReverseMap(Z_,z_[:,i],d,j)
        push!(err,norm(x-X[:,i])/norm(X[:,i]))
    end
    push!(e,Statistics.mean(err))
end
plot(e)
savefig("data/Figs/kPCA_PB_S10_Phi2000/ReverseMap_ErrInTSVSns.png")

X_test = []
for i in 1:3
    push!(X_test,CSV.File("data/csv/EM_PB_10S_Phi2000_test/EM_PB_10S_$i.csv") |> Tables.matrix)
end
X_test = hcat(X_test...)

conf_test = CSV.File("data/csv/EM_PB_10S_Phi2000_test/Config_N60_EM_PB_10S.csv") |> Tables.matrix

z_test = []
for c in eachcol(conf_test)
    z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
    push!(z_test,z)
end
z_test = reduce(hcat,z_test[[1:30...]])
scatter(eachrow(z_test)..., hover= [i for i in 1:lastindex(eachcol(z_test))])
savefig("data/Figs/kPCA_PB_S10_Phi2000/TestSet30_inRS_Optimized.png")

Λ_test, U_test, U__test, Ḡ_test = kPOD(Κ, X_test, k)
Z_test = real.(U_'*Ḡ_test)

n_test = 10
x, w_ns, Z_ns = ReverseMap(Z_,z_test[:,n_test],d,3)
plot(x)
plot!(X_test[:,n_test])
savefig("data/Figs/kPCA_PB_S10_Phi2000/ReverseMap_Xtest10_3Vecinos.png")
conf[:,n_test]
norm(x-X_test[:,n_test])/norm(X_test[:,n_test])

e = []
for j in 1:100
    err = []
    for i in 1:30
        x, w_ns, Z_ns = ReverseMap(Z_,z_test[:,i],d,j)
        push!(err,norm(x-X_test[:,i])/norm(X_test[:,i]))
    end
    push!(e,Statistics.mean(err))
end
plot(e)
savefig("data/Figs/kPCA_PB_S10_Phi2000/ReverseMap_ErrInTestSetVSns.png")

X_test

conf_complete = []
for x1 in 0:2, x2 in 0:2, x3 in 0:2, x4 in 0:2, x5 in 0:2, x6 in 0:2, x7 in 0:2, x8 in 0:2, x9 in 0:2, x10 in 0:2
  push!(conf_complete,[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
end
conf_complete = reduce(hcat,conf_complete)

z_complete = []
for c in eachcol(conf_complete)
    z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
    push!(z_complete,z)
end
z_complete = reduce(hcat,z_complete)
scatter(eachrow(z_complete)...,xlabel="z1",ylabel="z2",zlabel="z3",  
title=" Κ(X1,X2) = (X1'*X2 + β)^2<br>β = $β",hover= [i for i in 1:lastindex(eachcol(z_complete))])
savefig("data/Figs/kPCA_PB_S10_Phi2000/CompleteSet_inRS_Optimized.png")

X_complete = []
for z in eachcol(z_complete)
    x, w_ns, Z_ns = ReverseMap(Z_,z,d,3)
    push!(X_complete, x)
end

n_test = 10
Compare = []
for i in 1:lastindex(X_complete)
    push!(Compare,norm(X_complete[i]-X_test[:,n_test])/norm(X_test[:,n_test]))
end
minimum(Compare)
Compare_sort = sortperm(Compare, rev=false)
conf_test[:,10]
conf_complete[:,Compare_sort[1]]
for i in 1:lastindex(Compare_sort)
    if conf_test[:,10]==conf_complete[:,Compare_sort[i]]
        println(i)
    end
end
Compare[Compare_sort[23]]
plot!(X_complete[Compare_sort[23]])
plot(X_test[:,n_test])
x, w_ns, Z_ns = ReverseMap(Z_,z_test[:,n_test],d,3)
plot!(x)


n_test = 10
Compare = []
x_test, w_ns, Z_ns = ReverseMap(Z_,z_test[:,n_test],d,10)
for i in 1:lastindex(X_complete)
    push!(Compare,norm(X_complete[i]-x_test)/norm(x_test))
end
minimum(Compare)
Compare_sort = sortperm(Compare, rev=false)
conf_test[:,10]
conf_complete[:,Compare_sort[1]]

=#
