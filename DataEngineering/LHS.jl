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
Electrode = [[0.0,0.3],[0.0,0.3],[0.0,0.3],[0.0,0.3]] # Ranges of potential in the electrodes

function  electrode(number::Int,value::String)
    if value == "max"
        return maximum(Electrode[number])
    else 
        return minimum(Electrode[number])
    end
end

s1 = QuasiMonteCarlo.sample(points, [electrode(1,"max"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
s2 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"min"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
s3 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"max"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
s4 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"min"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
s5 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"max"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
s6 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"min"),electrode(4,"max")], LatinHypercubeSample())
s7 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"max")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"max")], LatinHypercubeSample())
s8 = QuasiMonteCarlo.sample(points, [electrode(1,"min"),electrode(2,"min"),electrode(3,"min"),electrode(4,"min")], [electrode(1,"max"),electrode(2,"max"),electrode(3,"max"),electrode(4,"min")], LatinHypercubeSample())

total = round.(hcat(s1,s2,s3,s4,s5,s6,s7,s8);sigdigits=4)

open("LHS.csv","w") do io
    writedlm(io,total)
end
