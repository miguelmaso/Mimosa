
macro publish(mod,name)
    quote
      using Mimosa.$mod: $name; export $name
    end
  end
  

  @publish TensorAlgebra (*)
  @publish TensorAlgebra cross_I4

  @publish FileManagement setupfolder
  @publish FileManagement numfiles
  @publish FileManagement _get_kwarg

  @publish ConstitutiveModels logreg
  @publish ConstitutiveModels NeoHookean3D
  @publish ConstitutiveModels DerivativeStrategy
  @publish ConstitutiveModels MoneyRivlin3D

   




  @publish LinearSolvers IterativeSolver
  # @publish LinearSolvers solve
  # @publish LinearSolvers solve!