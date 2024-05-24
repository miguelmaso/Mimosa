function  _TestFESpace(model, reffe, bc::DirichletBC, conf)
     TestFESpace(model, reffe, dirichlet_tags=bc.tags, conformity=conf)
end

function _TestFESpace(model, reffe, ::NothingBC, conf)
     TestFESpace(model, reffe, conformity=conf)
end
 
function _TrialFESpace(V, bc::DirichletBC, Λ)
    TrialFESpace(V, map(f -> f(Λ), bc.values))
end

function _TrialFESpace(V, ::NothingBC, Λ)
    V
end


# ========================
# ThermoMechProblem
# ========================

function get_FE_spaces(::ThermoMechProblem{:monolithic},
    model,
    order::Int64,
    bconds::MultiFieldBoundaryCondition; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeθ = ReferenceFE(lagrangian, Float64, 1)

    # Test FE Spaces
    Vu = _TestFESpace(model, reffeu, bconds.BoundaryCondition[1], :H1)
    Vθ = _TestFESpace(model, reffeθ, bconds.BoundaryCondition[2], :H1)

    # Trial FE Spaces
    Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],1.0)
    Uθ =  _TrialFESpace(Vθ,bconds.BoundaryCondition[2],1.0)

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vθ])
    U = MultiFieldFESpace([Uu, Uθ])

    return @ntuple Vu Vθ Uu Uθ V U
end

function get_FE_spaces!(::ThermoMechProblem{:monolithic},
    fe_spaces,
    bconds::MultiFieldBoundaryCondition, Λ)

    @unpack Vu, Vθ = fe_spaces

    # Trial FE Spaces
    Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],Λ)
    Uθ =  _TrialFESpace(Vθ,bconds.BoundaryCondition[2],Λ)
    
    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vθ])
    U = MultiFieldFESpace([Uu, Uθ])

    fe_spaces = @ntuple Vu Vθ Uu Uθ V U
end


# ===================
# ElectroMechProblem
# ===================

function get_FE_spaces(::ElectroMechProblem{:monolithic},
    model,
    order::Int64,
    bconds::MultiFieldBoundaryCondition; constraint=nothing)


    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)
 
    # Test FE Spaces
    Vu = _TestFESpace(model, reffeu, bconds.BoundaryCondition[1], :H1)
    Vφ = _TestFESpace(model, reffeφ, bconds.BoundaryCondition[2], :H1)

    # Trial FE Spaces
    Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],1.0)
    Uφ =  _TrialFESpace(Vφ,bconds.BoundaryCondition[2],1.0)

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([Uu, Uφ])

    return @ntuple Vu Vφ Uu Uφ V U
end

function get_FE_spaces!(::ElectroMechProblem{:monolithic},
    fe_spaces,
    bconds::MultiFieldBoundaryCondition, Λ)

    @unpack Vu, Vφ = fe_spaces

    # Trial FE Spaces
    Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],Λ)
    Uφ =  _TrialFESpace(Vφ,bconds.BoundaryCondition[2],Λ)

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ])
    U = MultiFieldFESpace([Uu, Uφ])

    fe_spaces = @ntuple Vu Vφ Uu Uφ V U
end

# ========================
# ThermoElectroMechProblem
# ========================

function get_FE_spaces(::ThermoElectroMechProblem{:monolithic},
    model,
    order::Int64,
    bconds::MultiFieldBoundaryCondition; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)
    reffeθ = ReferenceFE(lagrangian, Float64, 1)

    # Test FE Spaces
    Vu = _TestFESpace(model, reffeu, bconds.BoundaryCondition[1], :H1)
    Vφ = _TestFESpace(model, reffeφ, bconds.BoundaryCondition[2], :H1)
    Vθ = _TestFESpace(model, reffeθ, bconds.BoundaryCondition[3], :H1)

    # Trial FE Spaces
    Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],1.0)
    Uφ =  _TrialFESpace(Vφ,bconds.BoundaryCondition[2],1.0)
    Uθ =  _TrialFESpace(Vθ,bconds.BoundaryCondition[3],1.0)

    # Multifield FE Spaces
    V = MultiFieldFESpace([Vu, Vφ, Vθ])
    U = MultiFieldFESpace([Uu, Uφ, Uθ])

    return @ntuple Vu Vφ Vθ Uu Uφ Uθ V U
end

function get_FE_spaces!(::ThermoElectroMechProblem{:monolithic},
    fe_spaces,
    bconds::MultiFieldBoundaryCondition, Λ)

    @unpack Vu, Vφ, Vθ = fe_spaces

    # Trial FE Spaces
    Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],Λ)
    Uφ =  _TrialFESpace(Vφ,bconds.BoundaryCondition[2],Λ)
    Uθ =  _TrialFESpace(Vθ,bconds.BoundaryCondition[3],Λ)
    
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
    bconds::DirichletBC; constraint=nothing)

    # Reference FE
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    # Test FE Spaces
    V = _TestFESpace(model, reffeu, bconds, :H1)

    # Trial FE Spaces
    U =  _TrialFESpace(V,bconds,1.0)

    return @ntuple V U
end

function get_FE_spaces!(::MechanicalProblem,
    fe_spaces,
    bconds::DirichletBC, Λ)

    @unpack V = fe_spaces

    # Trial FE Spaces
    U =  _TrialFESpace(V,bconds,Λ)

    fe_spaces = @ntuple V U
end



