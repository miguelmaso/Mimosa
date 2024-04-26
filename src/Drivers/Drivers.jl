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

function ϝ(v::Float64)
  (x)->v
end

function ϝ(v::Vector{Float64})
  (x)->VectorValue(v)
end

function get_bc_func(tags_::Vector{String}, values_)
  bc_func_ =Vector{Function}(undef, length(tags_))
  @inbounds for i in eachindex(tags_)
      @assert(length(tags_) == length(values_))
           # get funcion generators for boundary conditions
              u_bc(Λ::Float64) = (x) -> ϝ(values_[i])(x) * Λ
              bc_func_[i] = u_bc
  end
   return (bc_tags=tags_, bc_func=bc_func_,)
end
 

function get_bc_func(tags_::Vector{Vector{String}}, values_)
  bc_func_ =Vector{Vector{Function}}(undef, length(tags_))
  @inbounds for i in eachindex(tags_)
      @assert(length(tags_[i]) == length(values_[i]))
        _, bc_func__ = get_bc_func(tags_[i], values_[i])
        bc_func_[i]=bc_func__

  end
   
  return (bc_tags=tags_, bc_func=bc_func_,)
end


function get_FE_solver(solveropt::Dict{Symbol,Real})
  nls_ = NLSolver(show_trace=solveropt[:nr_show_trace],
      method=:newton,
      iterations=solveropt[:nr_iter],
      ftol=solveropt[:nr_ftol])
  FESolver(nls_)
end


function Solver(problem::Problem, ctype::CouplingStrategy{:monolithic}, ph::FEFunction, params::Dict{Symbol,Any})

  nsteps = _get_kwarg(:nsteps, params[:solveropt])
  maxbisec = _get_kwarg(:nbisec, params[:solveropt])

  Λ = 0.0
  Λ_inc = 1.0 / nsteps

  cache = nothing
  nbisect = 0
  ph_view = get_free_dof_values(ph)

  while Λ < 1.0 - 1e-6
      Λ += Λ_inc
      Λ = min(1.0, Λ)
      ph_ = copy(get_free_dof_values(ph))
      ph, cache = ΔSolver!(problem, ctype, ph, Λ, Λ_inc, params, cache)
      flag = (cache.result.f_converged || cache.result.x_converged)

      #Check convergence
      if (flag == true)
          Λstring = replace("$Λ", "." => "_")

          # writevtk(Ω, "./results/result_" * Λstring, cellfields=["uh" => ph])
      else

          ph_view[:] = ph_
          # go back to previous ph
          Λ -= Λ_inc
          Λ_inc = Λ_inc / 2
          nbisect += 1
      end

      @assert(nbisect <= maxbisec, "Maximum number of bisections reached")

  end
  return ph
end


# Output function
function writePVD(filePath::String, trian::Triangulation, sol; append=false)
  outfiles = paraview_collection(filePath, append=append) do pvd
      for (i, (xh, t)) in enumerate(sol)
          println("STEP: $i, Lambda: $t")
          println("============================")
          uh = xh[1]
          vh = xh[2]
          ph = xh[3]
          pvd[t] = createvtk(
              trian,
              filePath * "_$t.vtu",
              cellfields = ["uh" => uh, "vh" => vh, "ph" => ph]
          )
      end
  end
end


include("FESpaces.jl")
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