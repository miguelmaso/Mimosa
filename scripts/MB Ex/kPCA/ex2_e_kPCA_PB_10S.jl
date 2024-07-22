n_p, P, d = 7, [2000], false
# Training Set in the full-order space
X = ReadData(n_p, P, d)
n = length(eachcol(X))
# Number of principal directions to be considered
k = 3
#Generation of full set of parameters in order
conf = CSV.File("data/csv/EM_PB_10S_Phi2000/Config_N1422_EM_PB_10S.csv") |> Tables.matrix
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
scatter(eachrow(Y_[:,group[param1_test]])...,xlabel="y1",ylabel="y2",label="TS reduced for PP1=0")#,color=[c[1] for c in eachcol(conf)],legend=false)
scatter!([y_gen[1]],[y_gen[2]],label="VS for conf_test 10")
n = lastindex(eachcol(Y_))
Y_gen = []
for i in 1:n
    y_ = VectorSearch(Y_,conf[:,i])
    push!(Y_gen,y_)
end
Y_gen = reduce(hcat,Y_gen)
Err_TS_1 = norm(Y_gen-Y_)/norm(Y_)
# ICP: rot and trans to reduce Err_TS
function Objective6_(x,x_,X_)
    θ = x[1]
    α = x[2]
    rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
    scale = [α 0;0 α]
    x_ = rot*x_
    x_ = scale*x_
    return norm(X_-x_)/norm(X_)
end
Objective6(x) = Objective6_(x,Y_gen,Y_)
x0 = [0.0,1.0]
_min = optimize(Objective6, x0, NelderMead())
θ = _min.minimizer[1]
α = _min.minimizer[2]
rot = [cos(θ) -sin(θ);sin(θ) cos(θ)]
scale = [α 0;0 α]
Y_gen = rot*Y_gen
Y_gen = scale*Y_gen
Err_TS_2 = norm(Y_gen-Y_)/norm(Y_)
# Input of test configurations
X_test = []
for i in 1:3
    push!(X_test,CSV.File("data/csv/EM_PB_10S_Phi2000_test/EM_PB_10S_$i.csv") |> Tables.matrix)
end
X_test = hcat(X_test...)
conf_test = CSV.File("data/csv/EM_PB_10S_Phi2000_test/Config_N60_EM_PB_10S.csv") |> Tables.matrix
# PP->FOS
n_test = 10
y_gen = VectorSearch(Y_,conf_test[:,n_test])
y_gen = rot*y_gen
y_gen = scale*y_gen
p = plot(X_test[:,n_test],label="FOStest$n_test: $(conf_test[:,n_test])",legend=:bottomright)
Err = []
for nh in 5:5
    x_gen = []
    for j in 1:10
        param1 = []
        for i in 1:n
            push!(param1,conf[j,i])
        end
        sort_param1 = sortperm(param1)
        group_param = 0
        group = [[],[],[]]
        count = 1
        for i in 1:n
            if param1[sort_param1[i]]==group_param
                push!(group[count],sort_param1[i])
            else
                count += 1
                group_param += 1
                push!(group[count],sort_param1[i])
            end
        end
        param1_test = conf_test[j,n_test]
        param1_test += 1
        sec = [((10*(j-1)+1)):((10*j)+1)...]
        x_gen_j, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]][sec,:],
        Y_[:,group[param1_test]],y_gen,false,nh)

        push!(x_gen,x_gen_j)
    end
    for i in 2:10
        Δ = x_gen[i][begin]-x_gen[i-1][end]
        x_gen[i] = x_gen[i].-Δ
    end
    for i in 1:lastindex(x_gen)-1
        pop!(x_gen[i])
    end
    x_gen = reduce(append!,x_gen)
    p = plot!(x_gen_0,label="FOSgen for conf_test$n_test - Step 3 adj - Err = $(norm(X_test[:,n_test]-x_gen_0)/norm(X_test[:,n_test]))")
    # push!(Err,norm(X_test[:,n_test]-x_gen)/norm(X_test[:,n_test]))
end
display(p)
plot(Err, type=:bar)
argmin(Err)
minimum(Err)

nh = 1
x_gen = []
for j in 1:10
    param1 = []
    for i in 1:n
        push!(param1,conf[j,i])
    end
    sort_param1 = sortperm(param1)
    group_param = 0
    group = [[],[],[]]
    count = 1
    for i in 1:n
        if param1[sort_param1[i]]==group_param
            push!(group[count],sort_param1[i])
        else
            count += 1
            group_param += 1
            push!(group[count],sort_param1[i])
        end
    end
    param1_test = conf_test[j,n_test]
    param1_test += 1
    sec = [((10*(j-1)+1)):((10*j)+1)...]
    # if j==10
    #     append!(sec,101)
    # end
    x_gen_j, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]][sec,:],
    Y_[:,group[param1_test]],y_gen,false,nh)
    push!(x_gen,x_gen_j)
end
for i in 2:10
    Δ = x_gen[i][begin]-x_gen[i-1][end]
    x_gen[i] = x_gen[i].-Δ
    θ1 = atan((x_gen[i-1][end] - x_gen[i-1][end-1])/0.001)
    θ2 = atan((x_gen[i][begin+1] - x_gen[i][begin])/0.001)
    Δθ = θ1 - θ2
    display((Δθ*(180/pi)))
    rot = [cos(Δθ) -sin(Δθ); sin(Δθ) cos(Δθ)]
    display(rot)
    x_gen_i = x_gen[i]
    b = x_gen_i[1]
    x_gen_i = x_gen_i.- b
    x_ = [0:10...]
    x_ = x_./1000
    X_ = hcat(x_,x_gen_i)
    X_ = X_'
    plot(eachrow(X_)...)
    X_ = rot*X_
    plot!(eachrow(X_)...)
    Poly(z) = [1 z z^2 z^3]
    A = Poly.(X_[1,:])
    A = reduce(vcat,A)
    c = pinv(A)*X_[2,:]
    A = Poly.(x_)
    A = reduce(vcat,A)
    x_gen[i] = A*c
    display(plot!(x_,x_gen[i],title="$i"))
    x_gen[i] .+= b
end
for i in 1:lastindex(x_gen)-1
    pop!(x_gen[i])
end
x_gen_0 = reduce(append!,x_gen)
p = plot(X_test[:,n_test])
p = plot!(x_gen_0)
norm(X_test[:,n_test]-x_gen_0)/norm(X_test[:,n_test])
frac = (X_test[:,n_test]-x_gen_0)./x_gen_0
plot(frac)

Err30 = []
for n_test in 1:30
    nh = 6
    x_gen = []
    for j in 1:10
        param1 = []
        for i in 1:n
            push!(param1,conf[j,i])
        end
        sort_param1 = sortperm(param1)
        group_param = 0
        group = [[],[],[]]
        count = 1
        for i in 1:n
            if param1[sort_param1[i]]==group_param
                push!(group[count],sort_param1[i])
            else
                count += 1
                group_param += 1
                push!(group[count],sort_param1[i])
            end
        end
        param1_test = conf_test[j,n_test]
        param1_test += 1
        sec = [((10*(j-1)+1)):((10*j)+1)...]
        x_gen_j, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]][sec,:],
        Y_[:,group[param1_test]],y_gen,false,nh)
        push!(x_gen,x_gen_j)
    end
    for i in 2:10
        Δ = x_gen[i][begin]-x_gen[i-1][end]
        x_gen[i] = x_gen[i].-Δ
        θ1 = atan((x_gen[i-1][end] - x_gen[i-1][end-1])/0.001)
        θ2 = atan((x_gen[i][begin+1] - x_gen[i][begin])/0.001)
        Δθ = θ1 - θ2
        # display((Δθ*(180/pi)))
        rot = [cos(Δθ) -sin(Δθ); sin(Δθ) cos(Δθ)]
        # display(rot)
        x_gen_i = x_gen[i]
        b = x_gen_i[1]
        x_gen_i = x_gen_i.- b
        x_ = [0:10...]
        x_ = x_./1000
        X_ = hcat(x_,x_gen_i)
        X_ = X_'
        # plot(eachrow(X_)...)
        X_ = rot*X_
        # plot!(eachrow(X_)...)
        Poly(z) = [1 z z^2 z^3]
        A = Poly.(X_[1,:])
        A = reduce(vcat,A)
        c = pinv(A)*X_[2,:]
        A = Poly.(x_)
        A = reduce(vcat,A)
        x_gen[i] = A*c
        # display(plot!(x_,x_gen[i],title="$i"))
        x_gen[i] .+= b
    end
    for i in 1:lastindex(x_gen)-1
        pop!(x_gen[i])
    end
    x_gen = reduce(append!,x_gen)
    push!(Err30,norm(X_test[:,n_test]-x_gen)/norm(X_test[:,n_test]))
end
plot(Err30, type=:bar)
mean(Err30)
Err30nh = []
for nh in 1:25
    Err30 = []
    for n_test in 1:30
        # nh = 1
        x_gen = []
        for j in 1:10
            param1 = []
            for i in 1:n
                push!(param1,conf[j,i])
            end
            sort_param1 = sortperm(param1)
            group_param = 0
            group = [[],[],[]]
            count = 1
            for i in 1:n
                if param1[sort_param1[i]]==group_param
                    push!(group[count],sort_param1[i])
                else
                    count += 1
                    group_param += 1
                    push!(group[count],sort_param1[i])
                end
            end
            param1_test = conf_test[j,n_test]
            param1_test += 1
            sec = [((10*(j-1)+1)):((10*j)+1)...]
            x_gen_j, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]][sec,:],
            Y_[:,group[param1_test]],y_gen,false,nh)
            push!(x_gen,x_gen_j)
        end
        for i in 2:10
            Δ = x_gen[i][begin]-x_gen[i-1][end]
            x_gen[i] = x_gen[i].-Δ
            θ1 = atan((x_gen[i-1][end] - x_gen[i-1][end-1])/0.001)
            θ2 = atan((x_gen[i][begin+1] - x_gen[i][begin])/0.001)
            Δθ = θ1 - θ2
            # display((Δθ*(180/pi)))
            rot = [cos(Δθ) -sin(Δθ); sin(Δθ) cos(Δθ)]
            # display(rot)
            x_gen_i = x_gen[i]
            b = x_gen_i[1]
            x_gen_i = x_gen_i.- b
            x_ = [0:10...]
            x_ = x_./1000
            X_ = hcat(x_,x_gen_i)
            X_ = X_'
            # plot(eachrow(X_)...)
            X_ = rot*X_
            # plot!(eachrow(X_)...)
            Poly(z) = [1 z z^2 z^3]
            A = Poly.(X_[1,:])
            A = reduce(vcat,A)
            c = pinv(A)*X_[2,:]
            A = Poly.(x_)
            A = reduce(vcat,A)
            x_gen[i] = A*c
            # display(plot!(x_,x_gen[i],title="$i"))
            x_gen[i] .+= b
        end
        for i in 1:lastindex(x_gen)-1
            pop!(x_gen[i])
        end
        x_gen = reduce(append!,x_gen)
        push!(Err30,norm(X_test[:,n_test]-x_gen)/norm(X_test[:,n_test]))
    end
    push!(Err30nh,mean(Err30))
end
plot(Err30nh, type=:bar)

function ReverseMap2(y_gen,conf_gen,Y_,X,conf,nh,t)
    x_gen = []
    for j in 1:10
        param1 = []
        for i in 1:n
            push!(param1,conf[j,i])
        end
        sort_param1 = sortperm(param1)
        group_param = 0
        group = [[],[],[]]
        count = 1
        for i in 1:n
            if param1[sort_param1[i]]==group_param
                push!(group[count],sort_param1[i])
            else
                count += 1
                group_param += 1
                push!(group[count],sort_param1[i])
            end
        end
        param1_test = conf_gen[j]
        param1_test += 1
        sec = [((10*(j-1)+1)):((10*j)+1)...]
        # if j==10
        #     append!(sec,101)
        # end
        x_gen_j, w_ns, Z_ns = ReverseMap(X[:,group[param1_test]][sec,:],
        Y_[:,group[param1_test]],y_gen,false,nh)
        push!(x_gen,x_gen_j)
    end
    for i in 2:10
        Δ = x_gen[i][begin]-x_gen[i-1][end]
        x_gen[i] = x_gen[i].-Δ
        θ1 = atan((x_gen[i-1][end] - x_gen[i-1][end-1])/0.001)
        θ2 = atan((x_gen[i][begin+1] - x_gen[i][begin])/0.001)
        Δθ = θ1 - θ2
        # Δθ = Δθ*1.5
        # display((Δθ*(180/pi)))
        rot = [cos(Δθ) -sin(Δθ); sin(Δθ) cos(Δθ)]
        # display(rot)
        x_gen_i = x_gen[i]
        b = x_gen_i[1]
        x_gen_i = x_gen_i.- b
        x_ = [0:10...]
        x_ = x_./1000
        X_ = hcat(x_,x_gen_i)
        X_ = X_'
        # # Poly2(z) = [1 z z^2 z^3]
        # Poly2(z) = [1 z z^2]
        # A = Poly2.(X_[1,:])
        # A = reduce(vcat,A)
        # c = pinv(A)*X_[2,:]
        # # Poly_d(z) = [0 1 2*z 3*z^2]
        # Poly_d(z) = [0 1 2*z]
        # A = Poly_d(-t)
        # θ2 = A*c
        # Δθ = θ1 - θ2[1]
        # rot = [cos(Δθ) -sin(Δθ); sin(Δθ) cos(Δθ)]
        # plot(eachrow(X_)...)
        X_ = rot*X_
        # plot!(eachrow(X_)...)
        Poly(z) = [1 z z^2 z^3]
        A = Poly.(X_[1,:])
        A = reduce(vcat,A)
        c = pinv(A)*X_[2,:]
        A = Poly.(x_)
        A = reduce(vcat,A)
        x_gen[i] = A*c
        # display(plot!(x_,x_gen[i],title="$i"))
        x_gen[i] .+= b
    end
    for i in 1:lastindex(x_gen)-1
        pop!(x_gen[i])
    end
    x_gen_0 = reduce(append!,x_gen)
    x_gen_0 =  1.10145.*x_gen_0
    return x_gen_0
end
n_test = 30
conf_gen = conf_test[:,n_test]
y_gen = VectorSearch(Y_,conf_complete[:,idx_1[30]])#conf_test[:,n_test])
y_gen = rot*y_gen
y_gen = scale*y_gen
x_gen_0 = ReverseMap2(y_gen,conf_gen,Y_,X,conf,5,0.0)
p = plot(X_test[:,n_test])
p = plot!(x_gen_0)
norm(x_gen_0-X_test[:,n_test])/norm(X_test[:,n_test])
scatter(x_gen_0,X_test[:,n_test],xlabel="x_gen",ylabel="x_test",legend=false)
Poly2(z) = [1 z]
A = Poly2.(x_gen_0)
A = reduce(vcat,A)
c = pinv(A)*(X_test[:,n_test])

E_ = []
E_avr = []
C_ = []
for nh in 5:5
    E__ = 0
    for n_test in 1:30
        print("\rn_test = $n_test")
        y_gen = VectorSearch(Y_,conf_test[:,n_test])
        y_gen = rot*y_gen
        y_gen = scale*y_gen
        conf_gen = conf_test[:,n_test]
        x_gen_0 = ReverseMap2(y_gen,conf_gen,Y_,X,conf,nh,0.0)
        E__ += norm(x_gen_0-X_test[:,n_test])/norm(X_test[:,n_test])
        push!(E_,norm(x_gen_0-X_test[:,n_test])/norm(X_test[:,n_test]))
        Poly2(z) = [1 z]
        A = Poly2.(x_gen_0)
        A = reduce(vcat,A)
        c = pinv(A)*(X_test[:,n_test])
        push!(C_,c)
    end
    push!(E_avr,E__/30)
    E__ = 0
end
plot(E_,type=:bar,label="Error norm(Δx)/norm(x) over 30 test with 5 nh")
MeanError = Statistics.mean(E_)
MeanErr = [MeanError for i in 1:30]
ME = round(MeanError, digits=4)
plot!(MeanErr, linestyle=:dash, linewidth=4, label="Mean Err = $ME")
plot(E_avr,type=:bar)
C__ = reduce(hcat,C_)
plot(C__[1,:])
mean(C__[1,:])

function Objective9(t)
    E_ = 0
    println("-------    t = $t ---------")
    for n_test in 1:30
        print("\rn_test = $n_test")
        y_gen = VectorSearch(Y_,conf_test[:,n_test])
        y_gen = rot*y_gen
        y_gen = scale*y_gen
        conf_gen = conf_test[:,n_test]
        x_gen_0 = ReverseMap2(y_gen,conf_gen,Y_,X,conf,nh,t)
        E_ += norm(x_gen_0-X_test[:,n_test])/norm(X_test[:,n_test])
    end
    return E_/30
end

t_min = optimize(Objective9, -0.01, 0.01, GoldenSection(),abs_tol=1.0e-8)