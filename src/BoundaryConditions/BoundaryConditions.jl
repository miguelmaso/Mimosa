module BoundaryConditions

using Gridap
using Gridap.TensorValues

export DirichletBC
export NeumannBC
export MultiFieldBoundaryCondition


function ϝ(v::Float64)
    (x) -> v
end

function ϝ(v::Vector{Float64})
    (x) -> VectorValue(v)
end

abstract type BoundaryCondition end

struct MultiFieldBoundaryCondition <: BoundaryCondition
    BoundaryCondition::Vector{BoundaryCondition}
end

struct DirichletBC <: BoundaryCondition
    tags::Vector{String}         # tags for boundary conditions
    values::Vector{Function}     # f(x)
    timesteps::Vector{Function}  # f(Λ)

    function DirichletBC(bc_tags::Vector{String}, bc_values, bc_timesteps)  
        @assert(length(bc_tags) == length(bc_values) == length(bc_timesteps))
        tags_,funcs_=_get_bc_func(bc_tags, bc_values, bc_timesteps)
        new(tags_, funcs_, bc_timesteps)
    end
end


struct NeumannBC <: BoundaryCondition
    tags::Vector{String}         # tags for boundary conditions
    values::Vector{Function}     # f(x)
    timesteps::Vector{Function}  # f(Λ)

    function NeumannBC(bc_tags::Vector{String}, bc_values, bc_timesteps)  
        @assert(length(bc_tags) == length(bc_values) == length(bc_timesteps))
        tags_,funcs_=_get_bc_func(bc_tags, bc_values, bc_timesteps)
        new(tags_, funcs_, bc_timesteps)
    end
end




function _get_bc_func(tags_::Vector{String}, values_,  bc_timesteps)
    bc_func_ = Vector{Function}(undef, length(tags_))
    @inbounds for i in eachindex(tags_)
        @assert(length(tags_) == length(values_))
        # get funcion generators for boundary conditions
        u_bc(Λ::Float64) = (x) -> ϝ(values_[i])(x) * bc_timesteps[i](Λ)
        bc_func_[i] = u_bc
    end
    return (bc_tags=tags_, bc_func=bc_func_,)
end








end