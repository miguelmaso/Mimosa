
macro publish(mod, name)
  quote
    using Mimosa.$mod: $name
    export $name
  end
end

@publish TensorAlgebra (*)
@publish TensorAlgebra (×ᵢ⁴)
@publish TensorAlgebra (⊗₁₂³)
@publish TensorAlgebra (⊗₁₃²)
@publish TensorAlgebra (⊗₁²³)
@publish TensorAlgebra (⊗₁₃²⁴)
@publish TensorAlgebra (⊗₁₂³⁴)
@publish TensorAlgebra (⊗₁²)
@publish TensorAlgebra logreg

@publish PhysicalModels DerivativeStrategy
@publish PhysicalModels LinearElasticity3D
@publish PhysicalModels NeoHookean3D
@publish PhysicalModels MoneyRivlin3D
@publish PhysicalModels ThermalModel
@publish PhysicalModels IdealDielectric
@publish PhysicalModels ElectroMechModel
@publish PhysicalModels ThermoElectroMechModel
@publish PhysicalModels ThermoMechModel
@publish PhysicalModels ThermoMech_EntropicPolyconvex

@publish PhysicalModels Mechano
@publish PhysicalModels Thermo
@publish PhysicalModels Electro
@publish PhysicalModels ThermoMechano
@publish PhysicalModels ElectroMechano
@publish PhysicalModels ThermoElectro
@publish PhysicalModels ThermoElectroMechano

@publish WeakForms residual
@publish WeakForms jacobian

@publish BoundaryConditions  DirichletBC
@publish BoundaryConditions  NeumannBC
@publish BoundaryConditions  NothingBC
@publish BoundaryConditions  MultiFieldBoundaryCondition

@publish Drivers PhysicalProblem
@publish Drivers ElectroMechProblem
@publish Drivers MechanicalProblem
@publish Drivers ThermoElectroMechProblem
@publish Drivers execute
@publish Drivers get_problem

@publish Solvers IterativeSolver
# @publish LinearSolvers solve
# @publish LinearSolvers solve!