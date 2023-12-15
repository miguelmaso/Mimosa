using Mimosa
using Gridap
using Test


TensorId = TensorValue(1.0, 0.5, 0.5, 1.0)


F(∇u) = one(∇u) + ∇u
J(F) = det(F)

@test J(TensorId) == 0.75
@test logreg(J(TensorId); Threshold=0.01) == -0.2876820724517809

