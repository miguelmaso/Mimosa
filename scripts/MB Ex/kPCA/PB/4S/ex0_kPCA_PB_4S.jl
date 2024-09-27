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
        push!(X,CSV.File("data/csv/EM_PB_4S_Phi$j/EM_PB_4S_$i.csv") |> Tables.matrix)
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
    Z_ = U_'*(X'*X)
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

function ReverseMap(Z_,z,d)
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

function VectorSearch(U_,Ḡ,conf)
    Z_ = U_'*Ḡ
    z0 = Z_[:,end]
    z1 = Z_[:,end-1]
    z2 = Z_[:,end-3]
    z3 = Z_[:,end-9]
    z4 = Z_[:,end-27]
    v = [[], [], [], []]
    v[4] = z1-z0
    v[3] = z2-z0
    v[2] = z3-z0
    v[1] = z4-z0
    z = z0
    
    for i in 1:lastindex(conf)
        if conf[i]==2
            conf[i] = 0
        elseif conf[i]==0
            conf[i] = -1
        elseif conf[i]==1
            conf[i] = 1
        end
    end
    for i in 1:lastindex(conf)
        z = z + conf[i]*v[i]
    end
    return z
end

function VectorSearch(Z_,conf)
    z0 = Z_[:,end]
    z1 = Z_[:,end-1]
    z2 = Z_[:,end-3]
    z3 = Z_[:,end-9]
    z4 = Z_[:,end-27]
    z5 = Z_[:,end-2]
    z6 = Z_[:,end-6]
    z7 = Z_[:,end-18]
    z8 = Z_[:,end-54]
    v = [[], [], [], [], [], [], [], []]
    v[8] = z5-z0
    v[7] = z6-z0
    v[6] = z7-z0
    v[5] = z8-z0
    v[4] = z1-z0
    v[3] = z2-z0
    v[2] = z3-z0
    v[1] = z4-z0
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
    z0 = Z_[:,end]
    z1 = Z_[:,end-1]
    z2 = Z_[:,end-3]
    z3 = Z_[:,end-9]
    z4 = Z_[:,end-27]
    z5 = Z_[:,end-2]
    z6 = Z_[:,end-6]
    z7 = Z_[:,end-18]
    z8 = Z_[:,end-54]
    v = [[], [], [], [], [], [], [], []]
    v[8] = z5-z0
    v[7] = z6-z0
    v[6] = z7-z0
    v[5] = z8-z0
    v[4] = z1-z0
    v[3] = z2-z0
    v[2] = z3-z0
    v[1] = z4-z0
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
        # println(R)
        u_1 = ∇zᵤ(z[1],z[2])
        R = RotM(u_1,u_0)
        # u_0 = u_1
    end
    return z
end

# function VectorSearch(U_,Ḡ,conf)
#     l = length(conf)
#     Z_ = U_'*Ḡ
#     z = Z_[:,[end-l:end...]]
#     v = [z[:,1]-z[:,i] for i in 2:l+1]
#     z = z[:,1]
#     for i in 1:lastindex(conf)
#         if conf[i]==0
#             conf[i] = -1
#         elseif conf[i]==2
#             conf[i] = 0
#         elseif conf[i]==1
#             conf[i] = 1
#         end
#     end
#     for i in 1:length(conf)
#         z = z + conf[i]*v[i]
#     end
#     return z
# end

function Distance0(β)
    println(β)
    P = [2000] #,4000]
    n_parts = 3
    k = 3
    d = false
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    Z_ = real.(U_'*Ḡ)
    dist1 = sqrt(dot(real.(Z_[:,1]-Z_[:,3]),real.(Z_[:,1]-Z_[:,3])))
    dist2 = sqrt(dot(real.(Z_[:,end]-Z_[:,end-1]),real.(Z_[:,end]-Z_[:,end-1])))
    println(abs(dist1-dist2)/dist2)
    return abs(dist1-dist2)/dist2
    # return 1-(abs(maximum(Z_[2,:]))/abs(maximum(Z_[1,:])))
end

function Distance0(β)
    println(β)
    P = [2000] #,4000]
    n_parts = 3
    k = 2
    d = false
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    # X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d)
    X = ReadData(n_parts,P,d)
    Λ, U, U_, Ḡ = kPOD(Κ,X,k)
    Z_ = real.(U_'*Ḡ)
    Dif(i,j,k,l) = abs((Dist(Z_,k,l)-Dist(Z_,i,j))/Dist(Z_,i,j))
    final = lastindex(eachcol(Z_))
    D = Dif(final,final-9,final-27,final-36)
    return D
    # return 1-(abs(maximum(Z_[2,:]))/abs(maximum(Z_[1,:])))
end

function RotM(v1,v2)
    v = cross(v1,v2)
    vₓ = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
    c = dot(v1,v2)
    R = I + vₓ + (1/(1+c))*vₓ*vₓ
    return R
end

Dist(Z_,i1,i2) = sqrt(dot(real.(Z_[:,i1]-Z_[:,i2]),real.(Z_[:,i1]-Z_[:,i2])))

#=
Quadric(z) = [1 z[1]^2 z[2]^2 z[1] z[2]]
Z_ = real.(U_'*Ḡ)
Z_c = Z_[3,:]
Z_[3,:] = Z_[2,:]
Z_[2,:] = Z_c
A = [Quadric(Z) for Z in eachcol(Z_)]
A = reduce(vcat,A)
c = pinv(A)*Z_[3,:]



Z_[3,:] = -(Z_[3,:].-c[1])
Z_[2,:] = Z_[2,:].*sqrt(c[3])
Z_[1,:] = Z_[1,:].*sqrt(c[2])

Z_[2,:] = sqrt.(abs.(Z_[2,:]))
Z_[1,:] = sqrt.(abs.(Z_[1,:]))

z3(z1,z2) = c[1] + c[2]*z1^2 + c[3]*z2^2
z3(z1,z2) = z1^2 + z2^2
z3(z1,z2) = c[1] + c[2]*z1^2 + c[3]*z2^2 + c[4]*z1 + c[5]*z2
∇zᵤ(z1,z2) = (1/norm([2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]))*[2*c[2]*z1 + c[4], 2*c[3]*z2 + c[5], 1]

z1 = range(minimum(Z_[1,:]), stop=maximum(Z_[1,:]), length=100)
z2 = range(minimum(Z_[2,:]), stop=maximum(Z_[2,:]), length=100)
z1 = range(-0.09, 0.09, length=100)
z2 = range(-0.09, 0.09, length=100)
surface!(z1,z2,z3)
surface(z1,z2,z3)
SS_reg = sqrt(dot(Z_[3,:]-z3.(Z_[1,:],Z_[2,:]),Z_[3,:]-z3.(Z_[1,:],Z_[2,:])))
avr = sum(Z_[3,:])/length(Z_[3,:])
SS_tot = sqrt(dot(Z_[3,:].-avr,Z_[3,:].-avr))
R_sq = 1-SS_reg/SS_tot
R = sqrt(R_sq)



conf = []
for i in 0:2, j in 0:2, k in 0:2, l in 0:2
  push!(conf,[i,j,k,l])
end
z_ = []
for c in conf
    z = VectorSearch(Z_,c,∇zᵤ); z = real.(z)
    push!(z_,z)
end
z_ = reduce(hcat,z_)
scatter!(eachrow(z_)...)
norm(Z_-z_)/norm(Z_)

conf = []
for i in 0:2, j in 0:2, k in 0:2, l in 0:2
  push!(conf,[i,j,k,l])
end
z_ = []
for c in conf
    z = VectorSearch(Z_,c); z = real.(z)
    push!(z_,z)
end
z_ = reduce(hcat,z_)
scatter!(eachrow(z_)...)
norm(Z_-z_)/norm(Z_)

conf = []
for i in 0:2, j in 0:2, k in 0:2, l in 0:2
  push!(conf,[i,j,k,l])
end
for c in conf
    z = VectorSearch(Z_,c); z = real.(z); display(scatter!([z[1]],[z[2]]))#,[z[3]]))
end

scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))], markercolor = [c[1] for c in conf])

Dist(Z_,81,72)
Dist(Z_,54,45)
(Dist(Z_,54,45)-Dist(Z_,81,72))/Dist(Z_,81,72)
Dif(i,j,k,l) = (Dist(Z_,k,l)-Dist(Z_,i,j))/Dist(Z_,i,j)
Dif(81,80,1,3)

scatter(eachrow(Z_)...,xlabel="z1",ylabel="z2",zlabel="z3", 
hover= [i for i in 1:lastindex(eachcol(Z_))])

scatter(Z_[1,:],Z_[2,:],xlabel="z1",ylabel="z2", 
hover= [i for i in 1:lastindex(eachcol(Z_))])

scatter!(z_[1,:],z_[2,:],xlabel="z1",ylabel="z2", 
hover= [i for i in 1:lastindex(eachcol(Z_))])

x, w_ns, Z_ns = ReverseMap(Z_,z_[:,6],d)
plot(x)
plot!(x6)
norm(x-x6)/norm(x6)
plot(x6)

x, w_ns, Z_ns = ReverseMap(Z_,z_[:,8],d)
plot(x)
plot!(x8)
norm(x-x8)/norm(x8)

norm(x6-x8)/norm(x8)

P = [2000] #,4000]
n_parts = 3
k = 3
d = false

Κ(X1,X2) = exp(-9.253901e-02*(dot(X1-X2,X1-X2))); X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d);
Κ(X1,X2) = exp(-1.264652e+00*(dot(X1-X2,X1-X2))); X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d);
Κ(X1,X2) = (X1'*X2 + 4.758223e-02)^2; X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d);
Κ(X1,X2) = (X1'*X2 + 10)^2; X, U_, Ḡ = execute_kPOD(Κ,P,n_parts,k,d);
Κ(X1,X2) = (X1'*X2 + 3.1)^4; X, U_ = execute_kPOD(Κ,P,n_parts,k,d);

X, U_ = execute_PCA(P,n_parts,k,d)
execute_POD(P,n_parts,k,d)
X, U_ = execute_kPOD(Κ,P,n_parts,k,d);
Z_ = U_'*(X'*X)
z = -Z_[:,38]
x, w_ns, Z_ns = ReverseMap(real.(U_),X,z,d)
plot(x)
plot(-X[:,39-9])
scatter!([z[1]],[z[2]],[z[3]])
z = VectorSearch(U_,X,conf)
z = real.(z)
conf = [0,0,1,2]
conf = [2,1,1,1]; z = VectorSearch(U_,Ḡ,conf); z = real.(z); scatter!([z[1]],[z[2]],[z[3]])
error = sqrt(dot(-X[:,39-9]-x,-X[:,39-9]-x))
Distance0(3.0e-7)
using Optim
β_min = optimize(Distance0, 0, 10, GoldenSection())
DDD = []
for i in range(start=1e-1, step=1e-2, stop=1e1)
    push!(DDD,Distance0(i))
end
plot(range(start=1e-1, step=1e-2, stop=1e1),DDD)
for i in 1:10
    Κ(X1,X2) = exp(-i*(dot(X1-X2,X1-X2))); X, U_ = execute_kPOD(Κ,P,n_parts,k,d);
end
=#

x6 = [0.0004
0.000402034
0.000413126
0.000435147
0.00046883
0.000514287
0.0005713
0.00063965
0.000719169
0.000809688
0.000911095
0.001023315
0.001146274
0.001279918
0.00142421
0.001579111
0.001744587
0.001920612
0.002107158
0.002304195
0.0025117
0.002729646
0.002958005
0.00319675
0.003445854
0.003705287
0.003975019
0.004255022
0.004545262
0.004845707
0.005156327
0.005477086
0.005807949
0.006148885
0.006499858
0.006860834
0.007231782
0.007612671
0.00800347
0.00840416
0.00881472
0.009235142
0.009665429
0.010105596
0.010555687
0.011015756
0.011485882
0.011966168
0.012456623
0.012957163
0.013466703
0.013976242
0.014476781
0.014967232
0.015447513
0.015917632
0.016377692
0.016827768
0.017267916
0.017698176
0.018118561
0.018529072
0.018929693
0.019320398
0.019701159
0.020071933
0.020432664
0.020783301
0.021123772
0.021453977
0.021773818
0.022083164
0.022381846
0.022669733
0.022946771
0.023212799
0.023473236
0.023732472
0.023992231
0.024252458
0.024513113
0.024774095
0.025035329
0.025296759
0.02555833
0.025820007
0.026081765
0.026343582
0.026605442
0.026867336
0.027129254
0.027391189
0.027653137
0.027915096
0.028177061
0.028439032
0.028701007
0.028962985
0.029224966
0.029486948
0.029748931]

x8 = [0.0004
0.000402034
0.000413126
0.000435147
0.00046883
0.000514287
0.0005713
0.00063965
0.000719169
0.000809688
0.000911095
0.001023315
0.001146274
0.001279918
0.00142421
0.001579111
0.001744586
0.001920611
0.002107157
0.002304193
0.002511698
0.002729644
0.002958001
0.003196745
0.003445847
0.003705277
0.003975006
0.004255005
0.004545238
0.004845674
0.005156283
0.005477026
0.005807868
0.006148776
0.00649971
0.006860632
0.00723151
0.007612302
0.008002969
0.008403481
0.0088138
0.009233894
0.009663739
0.010103314
0.010552609
0.011011631
0.011480399
0.011958962
0.012447359
0.012945613
0.013453644
0.013965197
0.014477397
0.014989225
0.015500616
0.016011582
0.016522213
0.017032591
0.017542775
0.018052827
0.018562789
0.019072689
0.019582556
0.020092414
0.020602281
0.021112181
0.021622142
0.022132195
0.022642379
0.023152756
0.023663388
0.024174354
0.024685745
0.025197573
0.025709773
0.026221325
0.026729357
0.027227611
0.027716008
0.028194571
0.02866334
0.029122362
0.029571658
0.030011234
0.03044108
0.030861175
0.031271497
0.031672012
0.032062684
0.032443482
0.032814369
0.033175303
0.033526252
0.033867177
0.034198035
0.034518787
0.034829389
0.035129777
0.035419906
0.035699706
0.035969204]
