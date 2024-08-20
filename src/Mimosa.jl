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

function julia_main()::Cint
    println("Test 01")
    include("/scripts/MB Ex/PB_4S/ex0_stat_EM_PB_4S_SIUnits.jl")
    return 0
end

end
