module MimosaTests

using Test

@testset "MimosaTests" verbose=true begin
    
    @testset "ConstitutiveModels" verbose=true include("ConstitutiveModels/runtests.jl")

    @testset "TensorAlgebra" verbose=false include("TensorAlgebra/runtests.jl")

end  # @testset "MimosaTests"

end  # module MimosaTests
