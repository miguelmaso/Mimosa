
function ϝ(v::Float64)
    (x) -> v
end

function ϝ(v::Vector{Float64})
    (x) -> VectorValue(v)
end

function get_bc_func(tags_::Vector{String}, values_)
    bc_func_ = Vector{Function}(undef, length(tags_))
    @inbounds for i in eachindex(tags_)
        @assert(length(tags_) == length(values_))
        # get funcion generators for boundary conditions
        u_bc(Λ::Float64) = (x) -> ϝ(values_[i])(x) * Λ
        bc_func_[i] = u_bc
    end
    return (bc_tags=tags_, bc_func=bc_func_,)
end


function get_bc_func(tags_::Vector{Vector{String}}, values_)
    bc_func_ = Vector{Vector{Function}}(undef, length(tags_))
    @inbounds for i in eachindex(tags_)
        @assert(length(tags_[i]) == length(values_[i]))
        _, bc_func__ = get_bc_func(tags_[i], values_[i])
        bc_func_[i] = bc_func__

    end

    return (bc_tags=tags_, bc_func=bc_func_,)
end

# function incremental_dirichletbc()
# end