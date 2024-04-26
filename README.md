# Mimosa :construction: :construction: :construction: **Work in progress** :construction: :construction: :construction:

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jmartfrut.github.io/Mimosa.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jmartfrut.github.io/Mimosa.jl/dev/)
[![Build Status](https://github.com/jmartfrut/Mimosa.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jmartfrut/Mimosa.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jmartfrut/Mimosa.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jmartfrut/Mimosa.jl)

# **M**ultiphysics-informed **D**esign of **T**unable **S**mart **M**aterials

This is an application repository with a collection of drivers for the simulation of Thermo-Electro-Magneto-Mehcanical problems. It is based on [Gridap](https://github.com/gridap/Gridap.jl), a package for grid-based approximation of PDEs with Finite Element.

## Usage
First, include the main Mimosa module:
```julia
using Mimosa
```
Then, execute the `main()` function especifying the driver name and the optional arguments. By default the `ElectroMechanics` driver is called, with the default parameters corresponding to the EM_Plate test case. Other drivers can be called with the following sintax:
```julia
main(; problemName="<driverName>",kwargs...)
```
Currently the following drivers are implemented:
  problemName = "EMPlate" ptype = "ElectroMechanics"

  - `problemName = "EMPlate"  ptype = "ElectroMechanics"`: A monolithic Electro-Mechanical simulation.
  - `problemName = "TEM_StaticSquare"  ptype = "ThermoElectroMechanics"`: A monolithic Static Thermo-Electro-Mechanical simulation.
  - `problemName = "M_Plate"  ptype = "Mechanics"`: A Static hyperelastic simulation.



## Contributing
Contributions with the definition of new drivers and additional Multiphysical formulations are welcome. The repository is organized as follows:
  - `Mimosa.jl`: module with the main function and inclusion of submodules.
  - `Drivers.jl`: module with a list of user-defined drivers. Each driver must implement the `execute` function with the corresponding problem name (`<driverName>`). 
    ```julia
    function execute(problem::Problem{:<driverName>}; kwargs...)
      # user-defined driver
    end
    ```
  - `WeakForms.jl`: module with the definition of the weak forms for all required residuals and Jacobians.
  - `ConstitutiveModels.jl`: module with the definition of the constitutive models providing energy and derivatives used in the weakforms.
  - `TensorAlgebra.jl`: module with the advanced Tensor Algebra operations used to define multiphysical constitutive models.


# Project funded by:
 
- Grant PID2022-141957OA-C22 funded by MCIN/AEI/ 10.13039/501100011033  and by ''ERDF A way of making Europe''

- Grant PID2022-141957OB-C22 funded by MCIN/AEI/ 10.13039/501100011033  and by ''ERDF A way of making Europe''


 <p align="center"> 
&nbsp; &nbsp; &nbsp; &nbsp;
<img alt="Dark"
src="https://github.com/jmartfrut/Mimosa/blob/main/docs/imgs/aei.png" width="70%">
</p>
 