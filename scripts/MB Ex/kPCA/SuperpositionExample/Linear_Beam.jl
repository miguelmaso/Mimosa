function EIy_i(x,w,a,b)
    R = -w*(b-a)
    Mr = -w*(b-a)*(a+b)
    EIy = (Mr/2)*x^2 + (R/6)*x^3
    if x>a
        EIy = EIy - (w/24)*(x-a)^4
    elseif x>b
        EIy = EIy - (w/24)*(x-a)^4 + (w/24)*(x-b)^4 
    end
    return EIy
end

# A = collect(0.0:0.25:0.75)
# B = collect(0.25:0.25:1.0)
# w1 = [10,0,0,0]
# w2 = [-10,0,0,0]

function EIy(x,w1,w2,A,B)
    EIy_ = 0
    for i in 1:lastindex(A)
        EIy_ = EIy_ + EIy_i(x,w1[i],A[i],B[i]) + EIy_i(x,w2[i],A[i],B[i])
    end
    return EIy_
end

function run()
    A = collect(0.0:0.25:0.75)
    B = collect(0.25:0.25:1.0)
    w1 = [0,0,0,0]
    w2 = [0,0,0,0]
    x = collect(0:1e-2:1)
    C = []
    for i1 in 0:1, i2 in 0:1, i3 in 0:1, i4 in 0:1, i5 in 0:1, i6 in 0:1, i7 in 0:1, i8 in 0:1
        push!(C,[i1,i2,i3,i4,i5,i6,i7,i8])
    end
    C = reduce(hcat,C)
    X = []
    for c in eachcol(C)
        count = 1
        for c_i in c
            if isodd(count)
                if c_i==1
                    w1[Int(((count-1)/2)+1)] = 10
                end
            else
                if c_i==1
                    w2[Int((count)/2)] = -10
                end
            end
            count += 1
        end
        EIy_(x_) = EIy(x_,w1,w2,A,B)
        push!(X,EIy_.(x))
        w1 = [0,0,0,0]
        w2 = [0,0,0,0]
    end
    X = reduce(hcat,X)
    return X
end