using Gridap


@testset "outer" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  @test norm(A ⊗ B) == 0.0002984572833756952
end

@testset "cross" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  C = A ⊗ B
  @test norm(cross_I4(A)) == 0.033763886032268264
  @test norm(A × B) == 6.246230863488799e-5
  @test norm(C × B) == 2.4491455542698976e-6
  @test norm(B × C) == 1.104276381618298e-6
end

@testset "inner" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  C = A ⊗ B
  D = TensorValue([4.6 2.1 1.7 3.2 6.5 1.4 9.2 8.0 9.0;
    1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0;
    5.3 2.0 3.1 1.9 5.4 9.8 0.4 8.8 3.1] * 1e-3)
  E = VectorValue(1.0, 2.0, 3.0) * 1e-3
  @test norm(C ⊙ A) == 4.676298215469156e-6
  @test norm(D ⊙ E) == 0.00010313946868197451
  @test norm(D ⊙ A) == 0.0004509607632599537
end

@testset "sum" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  @test norm(A + B) == 0.03393449572337859
end

