module Drivers
using DrWatson

using Gridap
using Gridap.Helpers
using GridapGmsh
using TimerOutputs
using WriteVTK
using ..ConstitutiveModels
using ..WeakForms

export Problem
export ElectroMechProblem
export MechanicalProblem
export ThermoElectroMechProblem

export execute
export get_problem

abstract type Problem end
abstract type SinglePhysicalProblem <: Problem end
abstract type MultiPhysicalProblem <: Problem end

struct MechanicalProblem{Kind} <: SinglePhysicalProblem end
struct ElectroMechProblem{Kind} <: MultiPhysicalProblem end
struct ThermoElectroMechProblem{Kind} <: MultiPhysicalProblem end

function get_problem(problemName::String, kwargs)

  ptype = _get_kwarg(:ptype,kwargs,"Mechanics")
  if ptype == "ElectroMechanics"
    return ElectroMechProblem{Symbol(problemName)}()
  elseif ptype == "ThermoElectroMechanics"
    return ThermoElectroMechProblem{Symbol(problemName)}()
  elseif ptype == "Mechanics"
    return MechanicalProblem{Symbol(problemName)}()
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
  println("\e[31m|            Parameter$(repeat(" ", maxlenghtkey - 7)) | Value\e[0m")
  println("\e[31m__________________________________________________________________________________________\e[0m")
  for (key, value) in input
    println("\e[31m|            $(string(key))$(repeat(" ", maxlenghtkey - length(string(key)) + 1)) | $value\e[0m")
  end
  println("\e[31m__________________________________________________________________________________________\e[0m")
end


function get_FE_solver(solveropt::Dict{Symbol,Real})
  nls_ = NLSolver(show_trace=solveropt[:nr_show_trace],
      method=:newton,
      iterations=solveropt[:nr_iter],
      ftol=solveropt[:nr_ftol])
  FESolver(nls_)
end


function IncrementalSolver(problem::Problem, ctype::CouplingStrategy{:monolithic}, ph::FEFunction, params::Dict{Symbol,Any})

  nsteps     = _get_kwarg(:nsteps, params[:solveropt])
  maxbisec   = _get_kwarg(:nbisec, params[:solveropt])
  filePath   = _get_kwarg(:simdir_, params[:post_params])
  is_vtk     = _get_kwarg(:is_vtk, params[:post_params])
  post_params     = _get_kwarg(:post_params, params)

  pvd        = paraview_collection(filePath*"/Results", append=false)

  Λ = 0.0
  Λ_inc = 1.0 / nsteps

  cache = nothing
  nbisect = 0
  ph_view = get_free_dof_values(ph)
  Λ_ = 0
  while Λ < 1.0 - 1e-6
      Λ += Λ_inc
      Λ = min(1.0, Λ)
      ph_ = copy(get_free_dof_values(ph))
      ph, cache = ΔSolver!(problem, ctype, ph, Λ, Λ_inc, params, cache)
      flag = (cache.result.f_converged || cache.result.x_converged)

      #Check convergence
      if (flag == true)
          Λ_ +=1
          # Write to PVD
          pvd = computeOutputs!(problem, pvd, ph, Λ, Λ_, post_params)
      else
          ph_view[:] = ph_
          # go back to previous ph
          Λ -= Λ_inc
          Λ_inc = Λ_inc / 2
          nbisect += 1
      end

      @assert(nbisect <= maxbisec, "Maximum number of bisections reached")

  end
  if is_vtk
  vtk_save(pvd)
  end
  return ph
end




include("FESpaces.jl")
include("BoundaryCond.jl")

include("ElectroMech_Drivers.jl")
include("ThermoElectroMech_Drivers.jl")
include("Mech_Drivers.jl")




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