module Drivers
using DrWatson

using Gridap
using Gridap.Helpers
using GridapGmsh
using TimerOutputs
using WriteVTK
using ..ConstitutiveModels
using ..WeakForms
using ..BoundaryConditions

export Problem
export ElectroMechProblem
export MechanicalProblem
export ThermoElectroMechProblem

export execute
export get_problem

abstract type Problem end
abstract type SinglePhysicalProblem <: Problem end
abstract type MultiPhysicalProblem <: Problem end

struct MechanicalProblem{KindReg} <: SinglePhysicalProblem end
struct ElectroMechProblem{KindSol, KindReg} <: MultiPhysicalProblem end
struct ThermoElectroMechProblem{KindSol, KindReg} <: MultiPhysicalProblem end

function get_problem(kwargs)
  ptype = _get_kwarg(:ptype,kwargs,"ElectroMechanics")
  soltype = _get_kwarg(:soltype,kwargs,"monolithic")
  regtype = _get_kwarg(:regtype,kwargs,"statics")

  if ptype == "ElectroMechanics"
    return ElectroMechProblem{Symbol(soltype), Symbol(regtype)}()
  elseif ptype == "ThermoElectroMechanics"
    return ThermoElectroMechProblem{Symbol(soltype), Symbol(regtype)}()
  elseif ptype == "Mechanics"
    return MechanicalProblem{Symbol(regtype)}()
  else
    @notimplemented("The problem type: $ptype is not implemented")
  end
end

execute(problem::Problem; kwargs...) = @notimplemented("The driver for problem: $problem is not implemented")

function _get_kwarg(kwarg,kwargs)
  try
      return kwargs[kwarg]
  catch
      s = "The key-word argument $(kwarg) is mandatory in the $problem driver"
      error(s)
  end
end

function _get_kwarg(kwarg,kwargs,value)
  # get the field kwarg from dictionary kwargs otherwise return value
  try
      return kwargs[kwarg]
  catch
      return value
  end
end
 

function setupfolder(folder_path::String)
  if !isdir(folder_path)
    mkdir(folder_path)
  else
    rm(folder_path,recursive=true)
    mkdir(folder_path)
  end
end


function print_heading(input::Dict)
  pname = _get_kwarg(:pname, input)
  ptype = _get_kwarg(:ptype, input)
  println("\e[31m__________________________________________________________________________________________\e[0m")
  println("\e[31m|                                                                                         \e[0m")
  println("\e[31m|                                 MULTISIMO LAB                                           \e[0m")
  println("\e[31m|                                                                                          \e[0m")
  println("\e[31m|            Executing MIMOSA Driver for $ptype Problem                                    \e[0m")
  println("\e[31m|                                                                                         \e[0m")
  println("\e[31m|            Problem name $pname                                                          \e[0m")
  println("\e[31m|                                                                                         \e[0m")
  println("\e[31m|                                                                                         \e[0m")
  maxlenghtkey = maximum(key -> length(string(key)), keys(input))
  # println("\e[31m|            Parameter$(repeat(" ", maxlenghtkey - 7)) | Value\e[0m")
  # println("\e[31m__________________________________________________________________________________________\e[0m")
  for (key, value) in input
    println("\e[31m|            $(string(key))$(repeat(" ", maxlenghtkey - length(string(key)) + 1)) | $value\e[0m")
  end
  println("\e[31m__________________________________________________________________________________________\e[0m")
end

 


# General Tools
include("FESpaces.jl")
include("Solvers.jl")

# Physical drivers
include("ElectroMechanics/Monolithic_Statics.jl")
include("ElectroMechanics/Monolithic_Dynamics.jl")
include("ThermoElectroMechanics/Monolithic_Statics.jl")
include("Mechanics/Statics.jl")
include("Mechanics/Dynamics.jl")




# function numfiles(foldername::String)
#   files_and_dirs = readdir(foldername)  # reading files and directory
#   num::Int64 = 0
#   for i in files_and_dirs
#       fullpath = joinpath(foldername, i)  # join foldername with file/directory name
#       if isfile(fullpath)
#           num += 1
#       end
#   end
#   return num
# end


end