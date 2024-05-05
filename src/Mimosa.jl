module Mimosa

using TimerOutputs

export main

include("Solvers/Solvers.jl")
include("TensorAlgebra/TensorAlgebra.jl")
include("ConstitutiveModels/ConstitutiveModels.jl")
include("WeakForms/WeakForms.jl")
include("BoundaryConditions/BoundaryConditions.jl")
include("Drivers/Drivers.jl")
include("Exports.jl")


function main(; kwargs...)

    reset_timer!()

    # Setup problem
    problem = get_problem(kwargs)

    # Execute driver
    outputs = execute(problem; kwargs...)

    print_timer()
    println()

    return outputs

end



end
