module BoundaryConditions

using Gridap
using Gridap.TensorValues

export DirichletBC
export NeumannBC
export get_Neumann_dΓ
export NothingBC
export MultiFieldBoundaryCondition
export residual_Neumann

abstract type BoundaryCondition end
struct NothingBC<: BoundaryCondition end


struct MultiFieldBoundaryCondition <: BoundaryCondition
    BoundaryCondition::Vector{BoundaryCondition}
end


function ϝ(v::Float64)
    (x) -> v
end

function ϝ(v::Vector{Float64})
    (x) -> VectorValue(v)
end

function ϝ(v::Function)
    if Base.return_types(v, (Vector{Float64},)) <: Vector
        VectorValue ∘ v
    else
        v
    end
end

function ϝ(v::Vector{Function})
    v
end


function _get_bc_func(tags_::Vector{String}, values_,  bc_timesteps)
    bc_func_ = Vector{Function}(undef, length(tags_))
    @assert(length(tags_) == length(values_))

    @inbounds for i in eachindex(tags_)
        # get funcion generators for boundary conditions
        u_bc(Λ::Float64) = (x) -> ϝ(values_[i])(x) * bc_timesteps[i](Λ)
        bc_func_[i] = u_bc
    end
    return (bc_tags=tags_, bc_func=bc_func_,)
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

    function DirichletBC(bc_tags::Vector{String}, bc_values)
        @assert(length(bc_tags) == length(bc_values))
        bc_timesteps = fill((_) -> 1.0, length(bc_tags))
        tags, funcs = _get_bc_func(bc_tags, bc_values, bc_timesteps)
        new(tags, funcs, bc_timesteps)
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


#------------------------------------------------------------
#                   Neumann Boundary conditions residuals
#------------------------------------------------------------

function residual_Neumann(::NothingBC, kwargs ...) end

function residual_Neumann(bc::NeumannBC, v, dΓ,  Λ)
    bc_func_ = Vector{Function}(undef, length(bc.tags))
     for (i,f) in enumerate(bc.values)
        bc_func_[i]=(v)->∫(-1.0*(v⋅f(Λ)))dΓ[i]
     end
     return mapreduce(f -> f(v), +, bc_func_)
end

function residual_Neumann(bc::NeumannBC, v, dΓ,  Λ⁺, Λ⁻)
    bc_func_ = Vector{Function}(undef, length(bc.tags))
     for (i,f) in enumerate(bc.values)
        bc_func_[i]=(v)->(∫(-0.5*(v⋅f(Λ⁺)))dΓ[i]+∫(-0.5*(v⋅f(Λ⁻)))dΓ[i])
     end
     return mapreduce(f -> f(v), +, bc_func_)
end

#------------------------------------------------------------
#                   Neumann Boundary conditions measures
#------------------------------------------------------------
function get_Neumann_dΓ(model,::NothingBC,degree)
    Vector{Gridap.CellData.GenericMeasure}(undef, 1)
end
 
function get_Neumann_dΓ(model,bc::NeumannBC,degree)
    dΓ=Vector{Gridap.CellData.GenericMeasure}(undef, length(bc.tags))
    for i in 1:length(bc.tags)
        Γ= BoundaryTriangulation(model, tags=bc.tags[i])
        dΓ[i]= Measure(Γ, degree)
    end
    return dΓ
end

function get_Neumann_dΓ(model,bc::MultiFieldBoundaryCondition,degree::Int64)
    dΓ=Vector{Vector{Gridap.CellData.GenericMeasure}}(undef, length(bc.BoundaryCondition))
    for (i,bc_i) in enumerate(bc.BoundaryCondition)
        dΓ[i]= get_Neumann_dΓ(model,bc_i,degree)
    end
    return dΓ
end


function _get_bc_func(values, timefunc::Vector{Function})
    map((v, t) -> bc_func(Λ::Float64) = (x) -> ϝ(v)(x) * t(Λ), values, timefunc)
end


function _get_bc_func(values)
    map(v -> bc_func(Λ::Float64) = (x) -> ϝ(v)(x), values)
end

end