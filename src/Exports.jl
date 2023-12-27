
macro publish(mod,name)
    quote
      using Mimosa.$mod: $name; export $name
    end
  end
  
  @publish TensorAlgebra inner42
  @publish TensorAlgebra inner32
  @publish TensorAlgebra inner31
  @publish TensorAlgebra (*)

  @publish FileManagement setupfolder
  @publish FileManagement numfiles
  @publish FileManagement _get_kwarg

  @publish ConstitutiveModels logreg

  @publish LinearSolvers IterativeSolver
  @publish LinearSolvers solve
  @publish LinearSolvers solve!
