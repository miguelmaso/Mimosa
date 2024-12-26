# Sensitibity test for the size of the TS
using JLD2
using DrWatson

include("C:/Users/mjbarillas/Documents/GitHub/Mimosa/scripts/MB Ex/kPCA/TB/ex2_aa_kPCA_TB_ST4SL4.jl")
include("C:/Users/mjbarillas/Documents/GitHub/Mimosa/scripts/Scratch/BPSO.jl")
include("C:/Users/mjbarillas/Documents/GitHub/Mimosa/scripts/MB Ex/TB ST4 SL4/ex0_stat_EM_TubeBeam_St&Sl.jl")


function run()
    St, Sl, pot_list, N_rand = 4, 4, [2000,3000,4000,5000], 200
    β = 2.5917492241092597
    k=3
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    neighbors = 25
    conf_list_test = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
    conf_list_test = conf_list_test[:,[1000-9:1000...]];
    VS_Conf_list = [1:64...]
    Result = []
    for N_rand in [200:100:600...]
        t_o = time()
        println("TS size = $N_rand + 64")
        conf_list, X = ReadData(St,Sl,pot_list,N_rand)
        Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
        Z_ = real.(U_'*Ḡ)
        Y_, D_G_sym = isomap1(neighbors,Z_)
        x_gen_list = []
        Err = []
        for i in 1:10
            n_test = i
            nh = 1
            pot = 4500
            conf = conf_list_test[:,n_test]
            x_gen_interpol = ReverseMap6(conf,pot,Y_,X,conf_list,pot_list,nh,VS_Conf_list)
            _x = ReadData_i(4,4,conf,pot)
            err = norm(x_gen_interpol-_x)/norm(_x)
            push!(Err,err)
        end
        println("ReverseMap6 Err Done!")
        result_list = []
        for n_test in 1:10
            println("n_test = $n_test Optimization = 1")
            result_list_test = []
            p_list = []
            nh = 1
            # n_test = 2
            conf = conf_list_test[:,n_test]
            pot = 4500
            _x = ReadData_i(4,4,conf,pot)
            function Obj_2(α)
                function Obj_int(Φ)
                    x_gen_interpol = ReverseMap6(α,Φ,Y_,X,conf_list,pot_list,nh,VS_Conf_list)
                    Err = (1-Κ(x_gen_interpol,_x))^2
                    return Err
                end
                Opt_ = optimize(Obj_int, 2000, 5000, GoldenSection(),f_tol=1.0e-6,iterations=50)
                return Opt_.minimum
            end
            (particles,Initial_X,Obj,w,c1,c2,iterations) = (18*4, [0 for _ in 1:16],Obj_2,0.5,1.5,0.5,100)
            @time state = optimize_BPSO(particles, Initial_X, Obj, w, c1, c2, iterations);
            function Obj_int(Φ)
                x_gen_interpol = ReverseMap6(state.P_gbest,Φ,Y_,X,conf_list,pot_list,nh,VS_Conf_list)
                Err = (1-Κ(x_gen_interpol,_x))^2
                return Err
            end
            Opt_ = optimize(Obj_int, 2000, 5000, GoldenSection(),f_tol=1.0e-6,iterations=50,show_trace=false)
            Voltage = round(Opt_.minimizer,digits = 0)
            println("n_test = $n_test Optimization = 1 simulation = 1")
            ph, chache = main(; get_parameters(Voltage, state.P_gbest, St, Sl,CenterLine_)...)
            _x_test = ReadData_i_test_last(4,4,state.P_gbest,Voltage)
            push!(result_list_test,[1,state.P_gbest,Voltage,norm(_x_test-_x)/norm(_x),_x_test])
            push!(p_list,state.P_gbest)
            println("First Error = $(norm(_x_test-_x)/norm(_x))")
            for i in 2:10
                println("n_test = $n_test Optimization = $i")
                @time state = optimize_BPSO(particles, Initial_X, Obj, w, c1, c2, iterations)
                println("Already found : $(maximum(string(state.P_gbest) .==string.(p_list)))")
                if maximum(string(state.P_gbest) .==string.(p_list))
                    i_ = first(findall(x->x==state.P_gbest,p_list))
                    result_list_test[i_][1] += 1
                else
                    function Obj_int2(Φ)
                        x_gen_interpol = ReverseMap6(state.P_gbest,Φ,Y_,X,conf_list,pot_list,nh,VS_Conf_list)
                        Err = (1-Κ(x_gen_interpol,_x))^2
                        return Err
                    end
                    Opt_ = optimize(Obj_int2, 2000, 5000, GoldenSection(),f_tol=1.0e-6,iterations=50,show_trace=false)
                    Voltage = round(Opt_.minimizer,digits = 0)
                    ph, chache = main(; get_parameters(Voltage, state.P_gbest, St, Sl,CenterLine_)...)
                    println("n_test = $n_test Optimization = $i simulation = $(length(result_list_test))")
                    _x_test = ReadData_i_test_last(4,4,state.P_gbest,Voltage)
                    push!(result_list_test,[1,state.P_gbest,Voltage,norm(_x_test-_x)/norm(_x),_x_test])
                    push!(p_list,state.P_gbest)
                end
            end
            push!(result_list,result_list_test)

        end
        t_1 = time()-t_o
        println("time = $t_1")
        push!(Result,[t_1,Err,result_list])
        jldsave("data/Sensitibity_to_TSsize.jdl2";Result)
    end

    
end

function readErr1(file)
    R = load(file)
    R = R["Result"]
    res = [[R[i][3][j][1][4] for j in 1:10] for i in 1:5]
    return res
end

function run2()
    St, Sl, pot_list, N_rand = 4, 4, [2000,3000,4000,5000], 200
    β = 2.5917492241092597
    k=3
    Κ(X1,X2) = exp(-β*(dot(X1-X2,X1-X2)))
    neighbors = 25
    conf_list_test = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
    conf_list_test = conf_list_test[:,[1000-9:1000...]];
    VS_Conf_list = [1:64...]
    Result = []
    for N_rand in [200:100:600...]
        t_o = time()
        println("TS size = $N_rand + 64")
        conf_list, X = ReadData(St,Sl,pot_list,N_rand)
        Λ, U, U_, Ḡ, G = kPOD(Κ, X, k)
        Z_ = real.(U_'*Ḡ)
        Y_, D_G_sym = isomap1(neighbors,Z_)
        t_1 = time()-t_o
        nh_Err_list = []
        for nh in 1:5
            Err = []
            for n_test in 1:10
                nh = 1
                pot = 4500
                conf = conf_list_test[:,n_test]
                x_gen_interpol = ReverseMap6(conf,pot,Y_,X,conf_list,pot_list,nh,VS_Conf_list)
                _x = ReadData_i(4,4,conf,pot)
                err = norm(x_gen_interpol-_x)/norm(_x)
                push!(Err,err)
            end
            push!(nh_Err_list,Err)
        end
        println("time = $t_1")
        D = @dict t_1 Λ U U_ Ḡ G Z_ Y_ D_G_sym nh_Err_list
        println("ReverseMap6 Err Done!")
        jldsave("data/Sensitibity_to_TSsize_$N_rand+64.jdl2";D)
    end

    
end

function run3()
    St, Sl, pot_list = 4, 4, [2000,3000,4000,5000]
    conf_list_test = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
    conf_list_test = conf_list_test[:,[1000-9:1000...]];
    VS_Conf_list = [1:64...]
    for N_rand in [200:100:600...]
        println("TS size = $N_rand + 64")
        conf_list, X = ReadData(St,Sl,pot_list,N_rand)
        D = load("data/Sensitibity_to_TSsize_$N_rand+64.jdl2")
        D = D["D"]
        U = D[:U]
        t_1 = D[:t_1]
        Ḡ = D[:Ḡ]
        G = D[:G]
        Λ = D[:Λ]
        Y_ = D[:Y_]
        D_G_sym = D[:D_G_sym]
        U_ = D[:U_]
        Z_ = D[:Z_]
        nh_Err_list = []
        for nh in 1:15
            Err = []
            for n_test in 1:10
                pot = 4500
                conf = conf_list_test[:,n_test]
                x_gen_interpol = ReverseMap6(conf,pot,Y_,X,conf_list,pot_list,nh,VS_Conf_list)
                _x = ReadData_i(4,4,conf,pot)
                err = norm(x_gen_interpol-_x)/norm(_x)
                push!(Err,err)
            end
            push!(nh_Err_list,Err)
        end
        D = @dict t_1 Λ U U_ Ḡ G Z_ Y_ D_G_sym nh_Err_list
        println("ReverseMap6 Err Done!")
        jldsave("data/Sensitibity_to_TSsize_$N_rand+64.jdl2";D)
    end
end

function Err_BCRM(nh)
    E = []
    plot()
    for N_rand in [200:100:600...]
        D = load("data/Sensitibity_to_TSsize_$N_rand+64.jdl2")
        D = D["D"]
        global p = plot!(D[:nh_Err_list][nh],ylims=(0.0,0.4)) 
        push!(E,reduce(hcat,D[:nh_Err_list][nh]))       
    end
    display(p)
    return reduce(hcat,E')
end

function run4(N_rand_low)
    St, Sl, pot_list = 4, 4, [2000,3000,4000,5000]
    conf_list_test = CSV.File("data/csv/EM_TB_St4_Sl4_Phi2000_Random/EM_TB_ST4_SL4_ConfRand.csv") |> Tables.matrix
    conf_list_test = conf_list_test[:,[1000-9:1000...]];
    VS_Conf_list = [1:64...]
    nh_Err = []
    for N_rand in [N_rand_low:100:600...]
        println("TS size = $N_rand + 64")
        conf_list, X = ReadData(St,Sl,pot_list,N_rand)
        D = load("data/Sensitibity_to_TSsize_$N_rand+64.jdl2")
        D = D["D"]
        U = D[:U]
        t_1 = D[:t_1]
        Ḡ = D[:Ḡ]
        G = D[:G]
        Λ = D[:Λ]
        Y_ = D[:Y_]
        D_G_sym = D[:D_G_sym]
        U_ = D[:U_]
        Z_ = D[:Z_]
        nh_Err_list = []
        for nh in 1:15
            Err = []
            for n_test in 1:10
                pot = 4500
                conf = conf_list_test[:,n_test]
                x_gen_interpol = ReverseMap6(conf,pot,Y_,X,conf_list,pot_list,nh,VS_Conf_list)
                _x = ReadData_i(4,4,conf,pot)
                err = norm(x_gen_interpol-_x)/norm(_x)
                push!(Err,err)
            end
            push!(nh_Err_list,Err)
        end
        nh_Err_ = reduce(hcat,nh_Err_list)
        push!(nh_Err,nh_Err_)
        println("ReverseMap6 Err Done!")
    end
    return reduce(hcat,nh_Err)
end