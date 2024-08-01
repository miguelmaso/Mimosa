# ------------------------------------------------------------------
# This file will create the data necessary for the data engineering
# ------------------------------------------------------------------


# We will take a 4-part electrode. Hence, we will need a vector of size 4x1 to be the input for our data generation.
# We will perform the data engineering considering that there is a 4x2=8 faceted cube that will constitute our design
# space. For each of this faces (each of the potentials), we will fix the min a max value of each potential and perform an
# LHS on the remaining potentials.

# The output should be a 4xa matrix where a=nx4 and n is the number of sampling points for each face. The output will be
# a .txt file that can be accessed column-wise to extract the values of the potential for each electrode.


# Two packages are mainly used for LHS sampling: QuasiMonteCarlo.jl and LatinHypercubeSampling.jl

using QuasiMonteCarlo
using Plots
using DelimitedFiles

points = 25

function fix_max(number::Int)
    range_min::Vector{Float64} = []
    range_max::Vector{Float64} = []
    for i in 1:20
        if i==number
            val = 0.3
        else
            val = 0.0
        end
    push!(range_min,val)
    end
    for i in 1:20
        val = 0.3
        push!(range_max,val)
    end
        
    return range_min, range_max
end
function fix_min(number::Int)
    range_min::Vector{Float64} = []
    range_max::Vector{Float64} = []
    for i in 1:20
        val = 0.0
        push!(range_min,val)
    end
    
    for i in 1:20
        if i==number
            val = 0.0
        else
            val = 0.3
        end
    push!(range_max,val)
    end
    return range_min, range_max
end

# s1 = QuasiMonteCarlo.sample(points, [electrode(1,"max"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
# s2 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"min"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
# s3 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"max"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
# s4 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"min"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
# s5 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"max"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
# s6 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"min"),electrode(4,"max")], LatinHypercubeSample())
# s7 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"max")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
# s8 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"min")], LatinHypercubeSample())
 for electrode in 1:20
    fmax_range_min, fmax_range_max = fix_max(electrode)
    fmin_range_min, fmin_range_max = fix_min(electrode)
    s_fmax = QuasiMonteCarlo.sample(points,[fmax_range_min[i] for i in 1:length(fmax_range_min)],[fmax_range_max[i] for i in 1:length(fmax_range_max)],LatinHypercubeSample())
    s_fmin = QuasiMonteCarlo.sample(points,[fmin_range_min[i] for i in 1:length(fmin_range_min)],[fmin_range_max[i] for i in 1:length(fmin_range_max)],LatinHypercubeSample())
    name=Symbol("Electrode_$electrode")
    value = hcat(s_fmax,s_fmin)
    eval(:($name = $value))

end


total = round.(hcat(Electrode_1,Electrode_2,Electrode_3,Electrode_4,Electrode_5,Electrode_6,Electrode_7,Electrode_8,Electrode_9,Electrode_10,Electrode_11,Electrode_12,Electrode_13,Electrode_14,Electrode_15,Electrode_16,Electrode_17,Electrode_18,Electrode_19,Electrode_20,);sigdigits=4)

open("LHS_Complex.txt","w") do io
    writedlm(io,total)
end
