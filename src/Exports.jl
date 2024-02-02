
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

  @publish LinearSolvers IterativeSolver
  # @publish LinearSolvers solve
  # @publish LinearSolvers solve!