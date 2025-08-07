using Test

@testset "MechanicalModels" verbose=true include("MechanicalModelsTests.jl")

@testset "ThermalModels" verbose=true include("ThermalModelsTests.jl")
