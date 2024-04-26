# ===================
# ElectroMechProblem
# ===================

function get_FE_spaces(::ElectroMechProblem,
    ::CouplingStrategy{:monolithic},
    model,
    order::Int64,
    bconds; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)

    # Test FE Spaces
    Vu = TestFESpace(model, reffeu, dirichlet_tags=bconds.bc_tags[1], conformity=:H1)
    Vφ = TestFESpace(model, reffeφ, dirichlet_tags=bconds.bc_tags[2], conformity=:H1)

    # Trial FE Spaces
    Uu = TrialFESpace(Vu, map(f -> f(1.0), bconds.bc_func[1]))
    Uφ = TrialFESpace(Vφ, map(f -> f(1.0), bconds.bc_func[2]))

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([Uu, Uφ])

    return @ntuple Vu Vφ Uu Uφ V U
end

function get_FE_spaces!(::ElectroMechProblem,
    ::CouplingStrategy{:monolithic},
    fe_spaces,
    bconds, Λ)

    @unpack Vu, Vφ = fe_spaces

    # Trial FE Spaces
    Uu = TrialFESpace(Vu, map(f -> f(Λ), bconds.bc_func[1]))
    Uφ = TrialFESpace(Vφ, map(f -> f(Λ), bconds.bc_func[2]))

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([Uu, Uφ])

    fe_spaces = @ntuple Vu Vφ Uu Uφ V U
end

# ========================
# ThermoElectroMechProblem
# ========================

function get_FE_spaces(::ThermoElectroMechProblem,
    ::CouplingStrategy{:monolithic},
    model,
    order::Int64,
    bconds; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)
    reffeθ = ReferenceFE(lagrangian, Float64, 1)

    # Test FE Spaces
    Vu = TestFESpace(model, reffeu, dirichlet_tags=bconds.bc_tags[1], conformity=:H1)
    Vφ = TestFESpace(model, reffeφ, dirichlet_tags=bconds.bc_tags[2], conformity=:H1)
    Vθ = TestFESpace(model, reffeθ, dirichlet_tags=bconds.bc_tags[3], conformity=:H1)

    # Trial FE Spaces
    Uu = TrialFESpace(Vu, map(f -> f(1.0), bconds.bc_func[1]))
    Uφ = TrialFESpace(Vφ, map(f -> f(1.0), bconds.bc_func[2]))
    Uθ = TrialFESpace(Vθ, map(f -> f(1.0), bconds.bc_func[3]))

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ, Vθ])
    U = MultiFieldFESpace([Uu, Uφ, Uθ])

    return @ntuple Vu Vφ Vθ Uu Uφ Uθ V U
end

function get_FE_spaces!(::ThermoElectroMechProblem,
    ::CouplingStrategy{:monolithic},
    fe_spaces,
    bconds, Λ)

    @unpack Vu, Vφ, Vθ = fe_spaces

    # Trial FE Spaces
    Uu = TrialFESpace(Vu, map(f -> f(Λ), bconds.bc_func[1]))
    Uφ = TrialFESpace(Vφ, map(f -> f(Λ), bconds.bc_func[2]))
    Uθ = TrialFESpace(Vθ, map(f -> f(1.0), bconds.bc_func[3]))

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ, Vθ])
    U = MultiFieldFESpace([Uu, Uφ, Uθ])

    fe_spaces = @ntuple Vu Vφ Vθ Uu Uφ Uθ V U
end


# ========================
# MechanicalProblem
# ========================

function get_FE_spaces(::MechanicalProblem,
    model,
    order::Int64,
    bconds; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    # Test FE Spaces
    V = TestFESpace(model, reffeu, dirichlet_tags=bconds.bc_tags, conformity=:H1)

    # Trial FE Spaces
    U = TrialFESpace(V, map(f -> f(1.0), bconds.bc_func))

    return @ntuple V U
end

function get_FE_spaces!(::MechanicalProblem,
    fe_spaces,
    bconds, Λ)

    @unpack V = fe_spaces

    # Trial FE Spaces
    U = TrialFESpace(V, map(f -> f(Λ), bconds.bc_func))

    fe_spaces = @ntuple V U
end




@generated function residual_Neumann(model, tags_::Vector{String}, degree)
    str = ""
    for i in eachindex(tags_)
    Γ= BoundaryTriangulation(model, tags=tags_[i])
    dΓ= Measure(Γ, degree)
    str *= "∫(v⋅neumannbc[i]) dΓ"
    end
    Meta.parse(str)

end