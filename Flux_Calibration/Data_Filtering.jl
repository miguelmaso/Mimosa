using Flux
using Flux: train!
using Statistics
using Plots
using Zygote
using DelimitedFiles
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
x_train = readdlm("Input_Alberto_.txt", ',');
y_train = readdlm("Output_Alberto_.txt", ',');


plot(y_train, seriestype = :scatter, label=["Original Data"])