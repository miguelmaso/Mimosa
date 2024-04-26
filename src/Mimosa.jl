module Mimosa

using TimerOutputs

export main

include("Solvers/LinearSolvers.jl")
include("TensorAlgebra/TensorAlgebra.jl")
include("ConstitutiveModels/ConstitutiveModels.jl")
include("WeakForms/WeakForms.jl")
include("Drivers/Drivers.jl")
include("Exports.jl")


function main(; problemName::String="EM_Plate", kwargs...)

    reset_timer!()

    # Setup problem
    problem = get_problem(problemName, kwargs)

    # Execute driver
    outputs = execute(problem; kwargs...)

    print_timer()
    println()

    return outputs

end



end
