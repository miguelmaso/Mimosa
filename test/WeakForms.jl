using Gridap



@testset "Assembly Jacobian ThermoMechanics" begin

    modelMR = MoneyRivlin3D(3.0, 1.0, 2.0)
    modelT = ThermalModel(1.0, 1.0, 2.0)
    f(δθ::Float64)::Float64 = (δθ+1.0) / 1.0
    df(δθ::Float64)::Float64 = 1.0
    κ = 1.0
    modelTM = ThermoMech(modelT, modelMR, f, df)
    Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = modelTM(Mimosa.DerivativeStrategy{:analytic}())

    partition = (1, 1, 1)
    pmin = Point(0.0, 0.0, 0.0)
    L = VectorValue(1, 1, 1)
    pmax = pmin + L
    model = CartesianDiscreteModel(pmin, pmax, partition)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "support0", [1, 2, 3, 4, 5, 6, 7, 8])

    order = 1
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeθ = ReferenceFE(lagrangian, Float64, order)

    #Define test FESpaces
    Vu = TestFESpace(model, reffeu, labels=labels, conformity=:H1)
    Vθ = TestFESpace(model, reffeθ, labels=labels, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vθ])
    U = V
    #Setup integration
    degree = 2 * order
    Ωₕ = Triangulation(model)
    dΩ = Measure(Ωₕ, 2)

    uh = FEFunction(Vu, zeros(Float64, num_free_dofs(Vu)))
    θ(x) = x[1]-1.0
    θh = interpolate_everywhere(θ, Vθ)

    function jach(uh, θh)
        jac((du, dθ), (v, vθ)) = jacobian_TM(CouplingStrategy{:monolithic}(), (uh, θh), (du, dθ), (v, vθ), (∂Ψuu, ∂Ψuθ), κ, dΩ)
    end
    function jac_mech(uh, θh)
        jac(du, v) = jacobian_TM(CouplingStrategy{:staggered_M}(), (uh, θh), du, v, ∂Ψuu, dΩ)
    end
    function jac_termoh(uh, θh)
        jac(dθ, vθ) = jacobian_TM(CouplingStrategy{:staggered_T}(), dθ, vθ, κ, dΩ)
    end

    jac_ = assemble_matrix(jach(uh, θh), V, V)
    jac_m = assemble_matrix(jac_mech(uh, θh), Vu, Vu)
    jac_t = assemble_matrix(jac_termoh(uh, θh), Vθ, Vθ)


    @test norm(jac_) ≈  25.6044821500619
    @test jac_[1] ≈  1.5555555555555538
    @test jac_[end] ≈  0.33333333333333315
    @test norm(jac_m) ≈  24.92278198212218
    @test jac_m[1] ≈  1.5555555555555538
    @test jac_m[end] ≈  1.8333333333333321
    @test norm(jac_t) ≈  1.0540925533894592
    @test jac_t[1] ≈  0.33333333333333304
    @test jac_t[end] ≈  0.33333333333333315


end




@testset "Assembly Jacobian ElectroMechanics" begin
    modelMR = Mimosa.MoneyRivlin3D(3.0, 1.0, 2.0)
    modelID = Mimosa.IdealDielectric(4.0)
    modelelectro = Mimosa.ElectroMech(modelMR, modelID)

    Ψm, ∂Ψmu, ∂Ψmuu = modelMR(Mimosa.DerivativeStrategy{:analytic}())
    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelelectro(Mimosa.DerivativeStrategy{:analytic}())

    partition = (1, 1, 1)
    pmin = Point(0.0, 0.0, 0.0)
    L = VectorValue(1, 1, 1)
    pmax = pmin + L
    model = CartesianDiscreteModel(pmin, pmax, partition)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "support0", [1, 2, 3, 4, 5, 6, 7, 8])


    order = 1
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)

    #Define test FESpaces
    Vu = TestFESpace(model, reffeu, labels=labels, conformity=:H1)
    Vφ = TestFESpace(model, reffeφ, labels=labels, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    U = V
    #Setup integration
    degree = 2 * order
    Ωₕ = Triangulation(model)
    dΩ = Measure(Ωₕ, 2)

    uh = FEFunction(Vu, zeros(Float64, num_free_dofs(Vu)))
    φ(x) = x[1]
    φh = interpolate_everywhere(φ, Vφ)

    function jach(uh, φh)
        jac((du, dφ), (v, vφ)) = jacobian_EM(CouplingStrategy{:monolithic}(), (uh, φh), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)
    end
    function jac_mech(uh, φh)
        jac(du, v) = jacobian_EM(CouplingStrategy{:staggered_M}(), (uh, φh), du, v, ∂Ψuu, dΩ)
    end
    function jac_elech(uh, φh)
        jac(dφ, vφ) = jacobian_EM(CouplingStrategy{:staggered_E}(), (uh, φh), dφ, vφ, ∂Ψφφ, dΩ)
    end

    jac_ = assemble_matrix(jach(uh, φh), V, V)
    jac_m = assemble_matrix(jac_mech(uh, φh), Vu, Vu)
    jac_e = assemble_matrix(jac_elech(uh, φh), Vφ, Vφ)
≈
    @test norm(jac_) ≈ 18.934585248125135
    @test jac_[1] ≈ 0.7777777777777775
    @test jac_[end] ≈ -1.3333333333333326

    @test norm(jac_m) ≈ 16.420402846143244
    @test jac_m[1] ≈ 0.7777777777777775
    @test jac_m[end] ≈ 2.1111111111111094

    @test norm(jac_e) ≈ 4.216370213557837
    @test jac_e[1] ≈ -1.3333333333333321
    @test jac_e[end] ≈ -1.3333333333333326


end