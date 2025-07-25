module TensorAlgebra

using Gridap
using Gridap.TensorValues
using LinearAlgebra
using StaticArrays
import Base: *
import Base: +

export (*)
export (×ᵢ⁴)
export (+)
export (⊗₁₂³)
export (⊗₁₃²)
export (⊗₁²³)
export (⊗₁₃²⁴)
export (⊗₁₂³⁴)
export (⊗₁²)
export sqrtM
export Cofactor
export Cross_I4_A
export I3
export I9
export inner42
export Outer_12_34
export Outer_13_24
export Outer_14_23
export Contraction_IP_JPKL
export Contraction_IP_PJKL
export I3_

# outer ⊗ \otimes
# inner ⊙ \odot
# cross × \times
# sum +
# dot ⋅ * 


function _δδ_μ_2D(μ::Float64)
  TensorValue{4,4,Float64,16}(  
    2*μ,
    0.0,
    0.0,
    0.0,
    0.0,
    μ,
    μ,
    0.0,
    0.0,
    μ,
    μ,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0*μ)
end 

function _δδ_λ_2D(λ::Float64)
  TensorValue{4,4,Float64,16}(  
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ,
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ)
end 

function _δδ_μ_3D(μ::Float64)
  TensorValue{9,9,Float64,81}(  
    2.0*μ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    μ,
    0.0,
    μ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    μ,
    0.0,
    0.0,
    0.0,
    μ,
    0.0,
    0.0,
    0.0,
    μ,
    0.0,
    μ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0*μ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    μ,
    0.0,
    μ,
    0.0,
    0.0,
    0.0,
    μ,
    0.0,
    0.0,
    0.0,
    μ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    μ,
    0.0,
    μ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0*μ)
end

function _δδ_λ_3D(λ::Float64)
  TensorValue{9,9,Float64,81}(  
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ,
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ,
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ)
end





function Gridap.TensorValues.outer(A::TensorValue{D,D,Float64}, B::TensorValue{D,D,Float64}) where {D}
  return (A ⊗₁₂³⁴ B)
end

function Gridap.TensorValues.outer(A::VectorValue{D,Float64}, B::VectorValue{D,Float64}) where {D}
  return (A ⊗₁² B)
end

@generated function (⊗₁²)(A::VectorValue{D,Float64}, B::VectorValue{D,Float64}) where {D}
  str = ""
  for iB in 1:D
    for iA in 1:D
      str *= "A.data[$iA] * B.data[$iB], "
    end
  end
  Meta.parse("TensorValue{D,D, Float64}($str)")
end

@generated function (⊗₁₂³⁴)(A::TensorValue{D,D,Float64}, B::TensorValue{D,D,Float64}) where {D}
  str = ""
  for iB in 1:D*D
    for iA in 1:D*D
      str *= "A.data[$iA] * B.data[$iB], "
    end
  end
  Meta.parse("TensorValue{D*D,D*D, Float64}($str)")
end


function (⊗₁²³)(V::VectorValue{3,Float64},A::TensorValue{3,3,Float64})
  
  TensorValue{3,9,Float64,27}(A[1]*V[1],
  A[1]*V[2],
  A[1]*V[3],
  A[2]*V[1],
  A[2]*V[2],
  A[2]*V[3],
  A[3]*V[1],
  A[3]*V[2],
  A[3]*V[3],
  A[4]*V[1],
  A[4]*V[2],
  A[4]*V[3],
  A[5]*V[1],
  A[5]*V[2],
  A[5]*V[3],
  A[6]*V[1],
  A[6]*V[2],
  A[6]*V[3],
  A[7]*V[1],
  A[7]*V[2],
  A[7]*V[3],
  A[8]*V[1],
  A[8]*V[2],
  A[8]*V[3],
  A[9]*V[1],
  A[9]*V[2],
  A[9]*V[3])

end
function (⊗₁₂³)(A::TensorValue{3,3,Float64}, V::VectorValue{3,Float64})
  
  TensorValue{3,9,Float64,27}(A[1]*V[1],
  A[2]*V[1],
  A[3]*V[1],
  A[4]*V[1],
  A[5]*V[1],
  A[6]*V[1],
  A[7]*V[1],
  A[8]*V[1],
  A[9]*V[1],
  A[1]*V[2],
  A[2]*V[2],
  A[3]*V[2],
  A[4]*V[2],
  A[5]*V[2],
  A[6]*V[2],
  A[7]*V[2],
  A[8]*V[2],
  A[9]*V[2],
  A[1]*V[3],
  A[2]*V[3],
  A[3]*V[3],
  A[4]*V[3],
  A[5]*V[3],
  A[6]*V[3],
  A[7]*V[3],
  A[8]*V[3],
  A[9]*V[3])
end
  
function (⊗₁₃²)(A::TensorValue{3,3,Float64}, V::VectorValue{3,Float64})
  
  TensorValue{3,9,Float64,27}(A[1]*V[1],
  A[2]*V[1],
  A[3]*V[1],
  A[1]*V[2],
  A[2]*V[2],
  A[3]*V[2],
  A[1]*V[3],
  A[2]*V[3],
  A[3]*V[3],
  A[4]*V[1],
  A[5]*V[1],
  A[6]*V[1],
  A[4]*V[2],
  A[5]*V[2],
  A[6]*V[2],
  A[4]*V[3],
  A[5]*V[3],
  A[6]*V[3],
  A[7]*V[1],
  A[8]*V[1],
  A[9]*V[1],
  A[7]*V[2],
  A[8]*V[2],
  A[9]*V[2],
  A[7]*V[3],
  A[8]*V[3],
  A[9]*V[3])
end


  
function (⊗₁₃²⁴)(A::TensorValue{3,3,Float64}, B::TensorValue{3,3,Float64})

  TensorValue{9,9,Float64,81}(A[1]*B[1],
  A[2]*B[1],  
  A[3]*B[1],  
  A[1]*B[2],  
  A[2]*B[2],  
  A[3]*B[2],  
  A[1]*B[3],  
  A[2]*B[3],  
  A[3]*B[3],  
  A[4]*B[1],  
  A[5]*B[1],  
  A[6]*B[1],  
  A[4]*B[2],  
  A[5]*B[2],  
  A[6]*B[2],  
  A[4]*B[3],  
  A[5]*B[3],  
  A[6]*B[3],  
  A[7]*B[1],  
  A[8]*B[1],  
  A[9]*B[1],  
  A[7]*B[2],  
  A[8]*B[2],  
  A[9]*B[2],  
  A[7]*B[3],  
  A[8]*B[3],  
  A[9]*B[3],  
  A[1]*B[4],  
  A[2]*B[4],  
  A[3]*B[4],  
  A[1]*B[5],  
  A[2]*B[5],  
  A[3]*B[5],  
  A[1]*B[6],  
  A[2]*B[6],  
  A[3]*B[6],  
  A[4]*B[4],  
  A[5]*B[4],  
  A[6]*B[4],  
  A[4]*B[5],  
  A[5]*B[5],  
  A[6]*B[5],  
  A[4]*B[6], 
  A[5]*B[6],  
  A[6]*B[6],  
  A[7]*B[4],  
  A[8]*B[4],  
  A[9]*B[4],  
  A[7]*B[5],  
  A[8]*B[5],  
  A[9]*B[5],  
  A[7]*B[6],  
  A[8]*B[6],  
  A[9]*B[6],  
  A[1]*B[7],  
  A[2]*B[7],  
  A[3]*B[7],  
  A[1]*B[8],  
  A[2]*B[8],  
  A[3]*B[8],  
  A[1]*B[9],  
  A[2]*B[9],  
  A[3]*B[9],  
  A[4]*B[7],  
  A[5]*B[7],  
  A[6]*B[7],  
  A[4]*B[8],  
  A[5]*B[8],  
  A[6]*B[8],  
  A[4]*B[9],  
  A[5]*B[9],  
  A[6]*B[9],  
  A[7]*B[7],  
  A[8]*B[7],  
  A[9]*B[7],  
  A[7]*B[8],  
  A[8]*B[8],  
  A[9]*B[8],  
  A[7]*B[9],  
  A[8]*B[9],  
  A[9]*B[9])
end

function (×ᵢ⁴)(A::TensorValue{3,3,Float64})

  TensorValue(0.0, 0.0, 0.0, 0.0, A[9], -A[8], 0.0, -A[6], A[5], 0.0, 0.0, 0.0, -A[9],
    0.0, A[7], A[6], 0.0, -A[4], 0.0, 0.0, 0.0, A[8], -A[7], 0.0, -A[5], A[4], 0.0, 0.0, -A[9],
    A[8], 0.0, 0.0, 0.0, 0.0, A[3], -A[2], A[9], 0.0, -A[7], 0.0, 0.0, 0.0, -A[3], 0.0,
    A[1], -A[8], A[7], 0.0, 0.0, 0.0, 0.0, A[2], -A[1], 0.0, 0.0, A[6], -A[5], 0.0,
    -A[3], A[2], 0.0, 0.0, 0.0, -A[6], 0.0, A[4], A[3], 0.0, -A[1],
    0.0, 0.0, 0.0, A[5], -A[4], 0.0, -A[2], A[1], 0.0, 0.0, 0.0, 0.0)
end


function Gridap.TensorValues.cross(A::TensorValue{3,3,Float64}, B::TensorValue{3,3,Float64})

  TensorValue(A[5] * B[9] - A[6] * B[8] - A[8] * B[6] + A[9] * B[5],
    A[6] * B[7] - A[4] * B[9] + A[7] * B[6] - A[9] * B[4],
    A[4] * B[8] - A[5] * B[7] - A[7] * B[5] + A[8] * B[4],
    A[3] * B[8] - A[2] * B[9] + A[8] * B[3] - A[9] * B[2],
    A[1] * B[9] - A[3] * B[7] - A[7] * B[3] + A[9] * B[1],
    A[2] * B[7] - A[1] * B[8] + A[7] * B[2] - A[8] * B[1],
    A[2] * B[6] - A[3] * B[5] - A[5] * B[3] + A[6] * B[2],
    A[3] * B[4] - A[1] * B[6] + A[4] * B[3] - A[6] * B[1],
    A[1] * B[5] - A[2] * B[4] - A[4] * B[2] + A[5] * B[1])
end


function Gridap.TensorValues.cross(H::TensorValue{9,9,Float64}, A::TensorValue{3,3,Float64})

  TensorValue(A[9] * H[37] - A[8] * H[46] - A[6] * H[64] + A[5] * H[73],
    A[9] * H[38] - A[8] * H[47] - A[6] * H[65] + A[5] * H[74],
    A[9] * H[39] - A[8] * H[48] - A[6] * H[66] + A[5] * H[75],
    A[9] * H[40] - A[8] * H[49] - A[6] * H[67] + A[5] * H[76],
    A[9] * H[41] - A[8] * H[50] - A[6] * H[68] + A[5] * H[77],
    A[9] * H[42] - A[8] * H[51] - A[6] * H[69] + A[5] * H[78],
    A[9] * H[43] - A[8] * H[52] - A[6] * H[70] + A[5] * H[79],
    A[9] * H[44] - A[8] * H[53] - A[6] * H[71] + A[5] * H[80],
    A[9] * H[45] - A[8] * H[54] - A[6] * H[72] + A[5] * H[81],
    A[7] * H[46] - A[9] * H[28] + A[6] * H[55] - A[4] * H[73],
    A[7] * H[47] - A[9] * H[29] + A[6] * H[56] - A[4] * H[74],
    A[7] * H[48] - A[9] * H[30] + A[6] * H[57] - A[4] * H[75],
    A[7] * H[49] - A[9] * H[31] + A[6] * H[58] - A[4] * H[76],
    A[7] * H[50] - A[9] * H[32] + A[6] * H[59] - A[4] * H[77],
    A[7] * H[51] - A[9] * H[33] + A[6] * H[60] - A[4] * H[78],
    A[7] * H[52] - A[9] * H[34] + A[6] * H[61] - A[4] * H[79],
    A[7] * H[53] - A[9] * H[35] + A[6] * H[62] - A[4] * H[80],
    A[7] * H[54] - A[9] * H[36] + A[6] * H[63] - A[4] * H[81],
    A[8] * H[28] - A[7] * H[37] - A[5] * H[55] + A[4] * H[64],
    A[8] * H[29] - A[7] * H[38] - A[5] * H[56] + A[4] * H[65],
    A[8] * H[30] - A[7] * H[39] - A[5] * H[57] + A[4] * H[66],
    A[8] * H[31] - A[7] * H[40] - A[5] * H[58] + A[4] * H[67],
    A[8] * H[32] - A[7] * H[41] - A[5] * H[59] + A[4] * H[68],
    A[8] * H[33] - A[7] * H[42] - A[5] * H[60] + A[4] * H[69],
    A[8] * H[34] - A[7] * H[43] - A[5] * H[61] + A[4] * H[70],
    A[8] * H[35] - A[7] * H[44] - A[5] * H[62] + A[4] * H[71],
    A[8] * H[36] - A[7] * H[45] - A[5] * H[63] + A[4] * H[72],
    A[8] * H[19] - A[9] * H[10] + A[3] * H[64] - A[2] * H[73],
    A[8] * H[20] - A[9] * H[11] + A[3] * H[65] - A[2] * H[74],
    A[8] * H[21] - A[9] * H[12] + A[3] * H[66] - A[2] * H[75],
    A[8] * H[22] - A[9] * H[13] + A[3] * H[67] - A[2] * H[76],
    A[8] * H[23] - A[9] * H[14] + A[3] * H[68] - A[2] * H[77],
    A[8] * H[24] - A[9] * H[15] + A[3] * H[69] - A[2] * H[78],
    A[8] * H[25] - A[9] * H[16] + A[3] * H[70] - A[2] * H[79],
    A[8] * H[26] - A[9] * H[17] + A[3] * H[71] - A[2] * H[80],
    A[8] * H[27] - A[9] * H[18] + A[3] * H[72] - A[2] * H[81],
    A[9] * H[1] - A[7] * H[19] - A[3] * H[55] + A[1] * H[73],
    A[9] * H[2] - A[7] * H[20] - A[3] * H[56] + A[1] * H[74],
    A[9] * H[3] - A[7] * H[21] - A[3] * H[57] + A[1] * H[75],
    A[9] * H[4] - A[7] * H[22] - A[3] * H[58] + A[1] * H[76],
    A[9] * H[5] - A[7] * H[23] - A[3] * H[59] + A[1] * H[77],
    A[9] * H[6] - A[7] * H[24] - A[3] * H[60] + A[1] * H[78],
    A[9] * H[7] - A[7] * H[25] - A[3] * H[61] + A[1] * H[79],
    A[9] * H[8] - A[7] * H[26] - A[3] * H[62] + A[1] * H[80],
    A[9] * H[9] - A[7] * H[27] - A[3] * H[63] + A[1] * H[81],
    A[7] * H[10] - A[8] * H[1] + A[2] * H[55] - A[1] * H[64],
    A[7] * H[11] - A[8] * H[2] + A[2] * H[56] - A[1] * H[65],
    A[7] * H[12] - A[8] * H[3] + A[2] * H[57] - A[1] * H[66],
    A[7] * H[13] - A[8] * H[4] + A[2] * H[58] - A[1] * H[67],
    A[7] * H[14] - A[8] * H[5] + A[2] * H[59] - A[1] * H[68],
    A[7] * H[15] - A[8] * H[6] + A[2] * H[60] - A[1] * H[69],
    A[7] * H[16] - A[8] * H[7] + A[2] * H[61] - A[1] * H[70],
    A[7] * H[17] - A[8] * H[8] + A[2] * H[62] - A[1] * H[71],
    A[7] * H[18] - A[8] * H[9] + A[2] * H[63] - A[1] * H[72],
    A[6] * H[10] - A[5] * H[19] - A[3] * H[37] + A[2] * H[46],
    A[6] * H[11] - A[5] * H[20] - A[3] * H[38] + A[2] * H[47],
    A[6] * H[12] - A[5] * H[21] - A[3] * H[39] + A[2] * H[48],
    A[6] * H[13] - A[5] * H[22] - A[3] * H[40] + A[2] * H[49],
    A[6] * H[14] - A[5] * H[23] - A[3] * H[41] + A[2] * H[50],
    A[6] * H[15] - A[5] * H[24] - A[3] * H[42] + A[2] * H[51],
    A[6] * H[16] - A[5] * H[25] - A[3] * H[43] + A[2] * H[52],
    A[6] * H[17] - A[5] * H[26] - A[3] * H[44] + A[2] * H[53],
    A[6] * H[18] - A[5] * H[27] - A[3] * H[45] + A[2] * H[54],
    A[4] * H[19] - A[6] * H[1] + A[3] * H[28] - A[1] * H[46],
    A[4] * H[20] - A[6] * H[2] + A[3] * H[29] - A[1] * H[47],
    A[4] * H[21] - A[6] * H[3] + A[3] * H[30] - A[1] * H[48],
    A[4] * H[22] - A[6] * H[4] + A[3] * H[31] - A[1] * H[49],
    A[4] * H[23] - A[6] * H[5] + A[3] * H[32] - A[1] * H[50],
    A[4] * H[24] - A[6] * H[6] + A[3] * H[33] - A[1] * H[51],
    A[4] * H[25] - A[6] * H[7] + A[3] * H[34] - A[1] * H[52],
    A[4] * H[26] - A[6] * H[8] + A[3] * H[35] - A[1] * H[53],
    A[4] * H[27] - A[6] * H[9] + A[3] * H[36] - A[1] * H[54],
    A[5] * H[1] - A[4] * H[10] - A[2] * H[28] + A[1] * H[37],
    A[5] * H[2] - A[4] * H[11] - A[2] * H[29] + A[1] * H[38],
    A[5] * H[3] - A[4] * H[12] - A[2] * H[30] + A[1] * H[39],
    A[5] * H[4] - A[4] * H[13] - A[2] * H[31] + A[1] * H[40],
    A[5] * H[5] - A[4] * H[14] - A[2] * H[32] + A[1] * H[41],
    A[5] * H[6] - A[4] * H[15] - A[2] * H[33] + A[1] * H[42],
    A[5] * H[7] - A[4] * H[16] - A[2] * H[34] + A[1] * H[43],
    A[5] * H[8] - A[4] * H[17] - A[2] * H[35] + A[1] * H[44],
    A[5] * H[9] - A[4] * H[18] - A[2] * H[36] + A[1] * H[45])
end

function Gridap.TensorValues.cross(A::TensorValue{3,3,Float64}, H::TensorValue{9,9,Float64})

  TensorValue(A[5] * H[9] - A[6] * H[8] - A[8] * H[6] + A[9] * H[5],
    A[6] * H[7] - A[4] * H[9] + A[7] * H[6] - A[9] * H[4],
    A[4] * H[8] - A[5] * H[7] - A[7] * H[5] + A[8] * H[4],
    A[3] * H[8] - A[2] * H[9] + A[8] * H[3] - A[9] * H[2],
    A[1] * H[9] - A[3] * H[7] - A[7] * H[3] + A[9] * H[1],
    A[2] * H[7] - A[1] * H[8] + A[7] * H[2] - A[8] * H[1],
    A[2] * H[6] - A[3] * H[5] - A[5] * H[3] + A[6] * H[2],
    A[3] * H[4] - A[1] * H[6] + A[4] * H[3] - A[6] * H[1],
    A[1] * H[5] - A[2] * H[4] - A[4] * H[2] + A[5] * H[1],
    A[5] * H[18] - A[6] * H[17] - A[8] * H[15] + A[9] * H[14],
    A[6] * H[16] - A[4] * H[18] + A[7] * H[15] - A[9] * H[13],
    A[4] * H[17] - A[5] * H[16] - A[7] * H[14] + A[8] * H[13],
    A[3] * H[17] - A[2] * H[18] + A[8] * H[12] - A[9] * H[11],
    A[1] * H[18] - A[3] * H[16] - A[7] * H[12] + A[9] * H[10],
    A[2] * H[16] - A[1] * H[17] + A[7] * H[11] - A[8] * H[10],
    A[2] * H[15] - A[3] * H[14] - A[5] * H[12] + A[6] * H[11],
    A[3] * H[13] - A[1] * H[15] + A[4] * H[12] - A[6] * H[10],
    A[1] * H[14] - A[2] * H[13] - A[4] * H[11] + A[5] * H[10],
    A[5] * H[27] - A[6] * H[26] - A[8] * H[24] + A[9] * H[23],
    A[6] * H[25] - A[4] * H[27] + A[7] * H[24] - A[9] * H[22],
    A[4] * H[26] - A[5] * H[25] - A[7] * H[23] + A[8] * H[22],
    A[3] * H[26] - A[2] * H[27] + A[8] * H[21] - A[9] * H[20],
    A[1] * H[27] - A[3] * H[25] - A[7] * H[21] + A[9] * H[19],
    A[2] * H[25] - A[1] * H[26] + A[7] * H[20] - A[8] * H[19],
    A[2] * H[24] - A[3] * H[23] - A[5] * H[21] + A[6] * H[20],
    A[3] * H[22] - A[1] * H[24] + A[4] * H[21] - A[6] * H[19],
    A[1] * H[23] - A[2] * H[22] - A[4] * H[20] + A[5] * H[19],
    A[5] * H[36] - A[6] * H[35] - A[8] * H[33] + A[9] * H[32],
    A[6] * H[34] - A[4] * H[36] + A[7] * H[33] - A[9] * H[31],
    A[4] * H[35] - A[5] * H[34] - A[7] * H[32] + A[8] * H[31],
    A[3] * H[35] - A[2] * H[36] + A[8] * H[30] - A[9] * H[29],
    A[1] * H[36] - A[3] * H[34] - A[7] * H[30] + A[9] * H[28],
    A[2] * H[34] - A[1] * H[35] + A[7] * H[29] - A[8] * H[28],
    A[2] * H[33] - A[3] * H[32] - A[5] * H[30] + A[6] * H[29],
    A[3] * H[31] - A[1] * H[33] + A[4] * H[30] - A[6] * H[28],
    A[1] * H[32] - A[2] * H[31] - A[4] * H[29] + A[5] * H[28],
    A[5] * H[45] - A[6] * H[44] - A[8] * H[42] + A[9] * H[41],
    A[6] * H[43] - A[4] * H[45] + A[7] * H[42] - A[9] * H[40],
    A[4] * H[44] - A[5] * H[43] - A[7] * H[41] + A[8] * H[40],
    A[3] * H[44] - A[2] * H[45] + A[8] * H[39] - A[9] * H[38],
    A[1] * H[45] - A[3] * H[43] - A[7] * H[39] + A[9] * H[37],
    A[2] * H[43] - A[1] * H[44] + A[7] * H[38] - A[8] * H[37],
    A[2] * H[42] - A[3] * H[41] - A[5] * H[39] + A[6] * H[38],
    A[3] * H[40] - A[1] * H[42] + A[4] * H[39] - A[6] * H[37],
    A[1] * H[41] - A[2] * H[40] - A[4] * H[38] + A[5] * H[37],
    A[5] * H[54] - A[6] * H[53] - A[8] * H[51] + A[9] * H[50],
    A[6] * H[52] - A[4] * H[54] + A[7] * H[51] - A[9] * H[49],
    A[4] * H[53] - A[5] * H[52] - A[7] * H[50] + A[8] * H[49],
    A[3] * H[53] - A[2] * H[54] + A[8] * H[48] - A[9] * H[47],
    A[1] * H[54] - A[3] * H[52] - A[7] * H[48] + A[9] * H[46],
    A[2] * H[52] - A[1] * H[53] + A[7] * H[47] - A[8] * H[46],
    A[2] * H[51] - A[3] * H[50] - A[5] * H[48] + A[6] * H[47],
    A[3] * H[49] - A[1] * H[51] + A[4] * H[48] - A[6] * H[46],
    A[1] * H[50] - A[2] * H[49] - A[4] * H[47] + A[5] * H[46],
    A[5] * H[63] - A[6] * H[62] - A[8] * H[60] + A[9] * H[59],
    A[6] * H[61] - A[4] * H[63] + A[7] * H[60] - A[9] * H[58],
    A[4] * H[62] - A[5] * H[61] - A[7] * H[59] + A[8] * H[58],
    A[3] * H[62] - A[2] * H[63] + A[8] * H[57] - A[9] * H[56],
    A[1] * H[63] - A[3] * H[61] - A[7] * H[57] + A[9] * H[55],
    A[2] * H[61] - A[1] * H[62] + A[7] * H[56] - A[8] * H[55],
    A[2] * H[60] - A[3] * H[59] - A[5] * H[57] + A[6] * H[56],
    A[3] * H[58] - A[1] * H[60] + A[4] * H[57] - A[6] * H[55],
    A[1] * H[59] - A[2] * H[58] - A[4] * H[56] + A[5] * H[55],
    A[5] * H[72] - A[6] * H[71] - A[8] * H[69] + A[9] * H[68],
    A[6] * H[70] - A[4] * H[72] + A[7] * H[69] - A[9] * H[67],
    A[4] * H[71] - A[5] * H[70] - A[7] * H[68] + A[8] * H[67],
    A[3] * H[71] - A[2] * H[72] + A[8] * H[66] - A[9] * H[65],
    A[1] * H[72] - A[3] * H[70] - A[7] * H[66] + A[9] * H[64],
    A[2] * H[70] - A[1] * H[71] + A[7] * H[65] - A[8] * H[64],
    A[2] * H[69] - A[3] * H[68] - A[5] * H[66] + A[6] * H[65],
    A[3] * H[67] - A[1] * H[69] + A[4] * H[66] - A[6] * H[64],
    A[1] * H[68] - A[2] * H[67] - A[4] * H[65] + A[5] * H[64],
    A[5] * H[81] - A[6] * H[80] - A[8] * H[78] + A[9] * H[77],
    A[6] * H[79] - A[4] * H[81] + A[7] * H[78] - A[9] * H[76],
    A[4] * H[80] - A[5] * H[79] - A[7] * H[77] + A[8] * H[76],
    A[3] * H[80] - A[2] * H[81] + A[8] * H[75] - A[9] * H[74],
    A[1] * H[81] - A[3] * H[79] - A[7] * H[75] + A[9] * H[73],
    A[2] * H[79] - A[1] * H[80] + A[7] * H[74] - A[8] * H[73],
    A[2] * H[78] - A[3] * H[77] - A[5] * H[75] + A[6] * H[74],
    A[3] * H[76] - A[1] * H[78] + A[4] * H[75] - A[6] * H[73],
    A[1] * H[77] - A[2] * H[76] - A[4] * H[74] + A[5] * H[73])
end

function Gridap.TensorValues.cross(A::TensorValue{3,9,Float64}, B::TensorValue{3,3,Float64})
 
  TensorValue{3,9,Float64,27}(A[13]*B[9] - A[16]*B[8] - A[22]*B[6] + A[25]*B[5],
  A[14]*B[9] - A[17]*B[8] - A[23]*B[6] + A[26]*B[5],
  A[15]*B[9] - A[18]*B[8] - A[24]*B[6] + A[27]*B[5],
  A[16]*B[7] - A[10]*B[9] + A[19]*B[6] - A[25]*B[4],
  A[17]*B[7] - A[11]*B[9] + A[20]*B[6] - A[26]*B[4],
  A[18]*B[7] - A[12]*B[9] + A[21]*B[6] - A[27]*B[4],
  A[10]*B[8] - A[13]*B[7] - A[19]*B[5] + A[22]*B[4],
  A[11]*B[8] - A[14]*B[7] - A[20]*B[5] + A[23]*B[4],
  A[12]*B[8] - A[15]*B[7] - A[21]*B[5] + A[24]*B[4],
    A[7]*B[8] - A[4]*B[9] + A[22]*B[3] - A[25]*B[2],
    A[8]*B[8] - A[5]*B[9] + A[23]*B[3] - A[26]*B[2],
    A[9]*B[8] - A[6]*B[9] + A[24]*B[3] - A[27]*B[2],
    A[1]*B[9] - A[7]*B[7] - A[19]*B[3] + A[25]*B[1],
    A[2]*B[9] - A[8]*B[7] - A[20]*B[3] + A[26]*B[1],
    A[3]*B[9] - A[9]*B[7] - A[21]*B[3] + A[27]*B[1],
    A[4]*B[7] - A[1]*B[8] + A[19]*B[2] - A[22]*B[1],
    A[5]*B[7] - A[2]*B[8] + A[20]*B[2] - A[23]*B[1],
    A[6]*B[7] - A[3]*B[8] + A[21]*B[2] - A[24]*B[1],
    A[4]*B[6] - A[7]*B[5] - A[13]*B[3] + A[16]*B[2],
    A[5]*B[6] - A[8]*B[5] - A[14]*B[3] + A[17]*B[2],
    A[6]*B[6] - A[9]*B[5] - A[15]*B[3] + A[18]*B[2],
    A[7]*B[4] - A[1]*B[6] + A[10]*B[3] - A[16]*B[1],
    A[8]*B[4] - A[2]*B[6] + A[11]*B[3] - A[17]*B[1],
    A[9]*B[4] - A[3]*B[6] + A[12]*B[3] - A[18]*B[1],
    A[1]*B[5] - A[4]*B[4] - A[10]*B[2] + A[13]*B[1],
    A[2]*B[5] - A[5]*B[4] - A[11]*B[2] + A[14]*B[1],
    A[3]*B[5] - A[6]*B[4] - A[12]*B[2] + A[15]*B[1])
end


function Gridap.TensorValues.inner(Ten1::TensorValue{9,9,Float64}, Ten2::TensorValue{3,3,Float64})
  TensorValue(Ten1[1] * Ten2[1] + Ten1[10] * Ten2[2] + Ten1[19] * Ten2[3] + Ten1[28] * Ten2[4] + Ten1[37] * Ten2[5] + Ten1[46] * Ten2[6] + Ten1[55] * Ten2[7] + Ten1[64] * Ten2[8] + Ten1[73] * Ten2[9],
    Ten1[2] * Ten2[1] + Ten1[11] * Ten2[2] + Ten1[20] * Ten2[3] + Ten1[29] * Ten2[4] + Ten1[38] * Ten2[5] + Ten1[47] * Ten2[6] + Ten1[56] * Ten2[7] + Ten1[65] * Ten2[8] + Ten1[74] * Ten2[9],
    Ten1[3] * Ten2[1] + Ten1[12] * Ten2[2] + Ten1[21] * Ten2[3] + Ten1[30] * Ten2[4] + Ten1[39] * Ten2[5] + Ten1[48] * Ten2[6] + Ten1[57] * Ten2[7] + Ten1[66] * Ten2[8] + Ten1[75] * Ten2[9],
    Ten1[4] * Ten2[1] + Ten1[13] * Ten2[2] + Ten1[22] * Ten2[3] + Ten1[31] * Ten2[4] + Ten1[40] * Ten2[5] + Ten1[49] * Ten2[6] + Ten1[58] * Ten2[7] + Ten1[67] * Ten2[8] + Ten1[76] * Ten2[9],
    Ten1[5] * Ten2[1] + Ten1[14] * Ten2[2] + Ten1[23] * Ten2[3] + Ten1[32] * Ten2[4] + Ten1[41] * Ten2[5] + Ten1[50] * Ten2[6] + Ten1[59] * Ten2[7] + Ten1[68] * Ten2[8] + Ten1[77] * Ten2[9],
    Ten1[6] * Ten2[1] + Ten1[15] * Ten2[2] + Ten1[24] * Ten2[3] + Ten1[33] * Ten2[4] + Ten1[42] * Ten2[5] + Ten1[51] * Ten2[6] + Ten1[60] * Ten2[7] + Ten1[69] * Ten2[8] + Ten1[78] * Ten2[9],
    Ten1[7] * Ten2[1] + Ten1[16] * Ten2[2] + Ten1[25] * Ten2[3] + Ten1[34] * Ten2[4] + Ten1[43] * Ten2[5] + Ten1[52] * Ten2[6] + Ten1[61] * Ten2[7] + Ten1[70] * Ten2[8] + Ten1[79] * Ten2[9],
    Ten1[8] * Ten2[1] + Ten1[17] * Ten2[2] + Ten1[26] * Ten2[3] + Ten1[35] * Ten2[4] + Ten1[44] * Ten2[5] + Ten1[53] * Ten2[6] + Ten1[62] * Ten2[7] + Ten1[71] * Ten2[8] + Ten1[80] * Ten2[9],
    Ten1[9] * Ten2[1] + Ten1[18] * Ten2[2] + Ten1[27] * Ten2[3] + Ten1[36] * Ten2[4] + Ten1[45] * Ten2[5] + Ten1[54] * Ten2[6] + Ten1[63] * Ten2[7] + Ten1[72] * Ten2[8] + Ten1[81] * Ten2[9])
end

function Gridap.TensorValues.inner(Ten1::TensorValue{3,9,Float64}, Ten2::TensorValue{3,3,Float64})
  VectorValue(Ten1[1] * Ten2[1] + Ten1[4] * Ten2[2] + Ten1[7] * Ten2[3] + Ten1[10] * Ten2[4] + Ten1[13] * Ten2[5] + Ten1[16] * Ten2[6] + Ten1[19] * Ten2[7] + Ten1[22] * Ten2[8] + Ten1[25] * Ten2[9],
    Ten1[2] * Ten2[1] + Ten1[5] * Ten2[2] + Ten1[8] * Ten2[3] + Ten1[11] * Ten2[4] + Ten1[14] * Ten2[5] + Ten1[17] * Ten2[6] + Ten1[20] * Ten2[7] + Ten1[23] * Ten2[8] + Ten1[26] * Ten2[9],
    Ten1[3] * Ten2[1] + Ten1[6] * Ten2[2] + Ten1[9] * Ten2[3] + Ten1[12] * Ten2[4] + Ten1[15] * Ten2[5] + Ten1[18] * Ten2[6] + Ten1[21] * Ten2[7] + Ten1[24] * Ten2[8] + Ten1[27] * Ten2[9])
end


function Gridap.TensorValues.inner(Ten1::TensorValue{3,9,Float64}, Ten2::VectorValue{3,Float64})
  TensorValue(Ten1[1] * Ten2[1] + Ten1[10] * Ten2[2] + Ten1[19] * Ten2[3],
    Ten1[2] * Ten2[1] + Ten1[11] * Ten2[2] + Ten1[20] * Ten2[3],
    Ten1[3] * Ten2[1] + Ten1[12] * Ten2[2] + Ten1[21] * Ten2[3],
    Ten1[4] * Ten2[1] + Ten1[13] * Ten2[2] + Ten1[22] * Ten2[3],
    Ten1[5] * Ten2[1] + Ten1[14] * Ten2[2] + Ten1[23] * Ten2[3],
    Ten1[6] * Ten2[1] + Ten1[15] * Ten2[2] + Ten1[24] * Ten2[3],
    Ten1[7] * Ten2[1] + Ten1[16] * Ten2[2] + Ten1[25] * Ten2[3],
    Ten1[8] * Ten2[1] + Ten1[17] * Ten2[2] + Ten1[26] * Ten2[3],
    Ten1[9] * Ten2[1] + Ten1[18] * Ten2[2] + Ten1[27] * Ten2[3])
end



@inline function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end

@inline function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end


@generated function (+)(A::TensorValue{D,D,Float64}, B::TensorValue{D,D,Float64}) where {D}
  str = ""
  for i in 1:D*D
    str *= "A.data[$i] + B.data[$i], "
  end
  Meta.parse("TensorValue{D,D, Float64}($str)")
end

const I3_ = diagm(vec(ones(3,1)))
const I9_ = diagm(vec(ones(9,1)))

function I3() 
  TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
end 

function I9()
TensorValue(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
end


"""
  EigDecomposition

Compose a 3x3 matrix from its eigenvalues and eigenvectors
"""
@inline function EigDecomposition(λ::SVector,n::SMatrix)
  SMatrix{3,3}(λ[1]*n[1:3]*n[1:3]' + λ[2]*n[4:6]*n[4:6]' + λ[3]*n[7:9]*n[7:9]')
end


"""
  sqrtM(A::SMatrix)::SMatrix

Compute the square root of a 3x3 matrix by means of eigen decomposition.

# Arguments
- `A::SMatrix`: the matrix to calculate the square root

# Returns
- `::SMatrix`: the squared root matrix
"""
@inline function sqrtM(A::SMatrix)
  λ, Q = eigen(A)
  EigDecomposition(λ, Q)
end


"""
  Cofactor(A::SMatrix)::SMatrix

Calculate the cofactor of a matrix.

# Arguments
- `A::SMatrix`: the matrix to calculate.

# Returns
- `SMatrix`: the cofactor matrix.
"""
@inline function Cofactor(A::SMatrix)
 return det(A)*inv(A')
end


"""
  Cross_I4_A

  Calculate the cross product of ... TODO: Need documentation
"""
@inline function Cross_I4_A(A::SMatrix)
  dim = size(A,1)
  return SMatrix{dim*dim,dim*dim}(
    0.0, 0.0, 0.0, 0.0, A[9], -A[8], 0.0, -A[6], A[5], 0.0, 0.0, 0.0, -A[9],
    0.0, A[7], A[6], 0.0, -A[4], 0.0, 0.0, 0.0, A[8], -A[7], 0.0, -A[5], A[4], 0.0, 0.0, -A[9],
    A[8], 0.0, 0.0, 0.0, 0.0, A[3], -A[2], A[9], 0.0, -A[7], 0.0, 0.0, 0.0, -A[3], 0.0,
    A[1], -A[8], A[7], 0.0, 0.0, 0.0, 0.0, A[2], -A[1], 0.0, 0.0, A[6], -A[5], 0.0,
    -A[3], A[2], 0.0, 0.0, 0.0, -A[6], 0.0, A[4], A[3], 0.0, -A[1],
    0.0, 0.0, 0.0, A[5], -A[4], 0.0, -A[2], A[1], 0.0, 0.0, 0.0, 0.0)
end


"""
  inner42(::TensorValue, ::TensorValue)::TensorValue

Compute the inner product of a fourth-order and a second-order tensor in a 3D space.

# Arguments
- `Ten1::TensorValue{9,9}`: A fourth-order tensor reppresented by a dim²xdim² matrix.
- `Ten2::TensorValue{3,3}`: A second-order tensor.

# Returns
- `::TensorValue{3,3}`: The inner product.
"""
function inner42(Ten1::TensorValue, Ten2::TensorValue)
  TensorValue(
    Ten1.data[1] * Ten2.data[1] + Ten1.data[10] * Ten2.data[2] + Ten1.data[19] * Ten2.data[3] + Ten1.data[28] * Ten2.data[4] + Ten1.data[37] * Ten2.data[5] + Ten1.data[46] * Ten2.data[6] + Ten1.data[55] * Ten2.data[7] + Ten1.data[64] * Ten2.data[8] + Ten1.data[73] * Ten2.data[9],
    Ten1.data[2] * Ten2.data[1] + Ten1.data[11] * Ten2.data[2] + Ten1.data[20] * Ten2.data[3] + Ten1.data[29] * Ten2.data[4] + Ten1.data[38] * Ten2.data[5] + Ten1.data[47] * Ten2.data[6] + Ten1.data[56] * Ten2.data[7] + Ten1.data[65] * Ten2.data[8] + Ten1.data[74] * Ten2.data[9],
    Ten1.data[3] * Ten2.data[1] + Ten1.data[12] * Ten2.data[2] + Ten1.data[21] * Ten2.data[3] + Ten1.data[30] * Ten2.data[4] + Ten1.data[39] * Ten2.data[5] + Ten1.data[48] * Ten2.data[6] + Ten1.data[57] * Ten2.data[7] + Ten1.data[66] * Ten2.data[8] + Ten1.data[75] * Ten2.data[9],
    Ten1.data[4] * Ten2.data[1] + Ten1.data[13] * Ten2.data[2] + Ten1.data[22] * Ten2.data[3] + Ten1.data[31] * Ten2.data[4] + Ten1.data[40] * Ten2.data[5] + Ten1.data[49] * Ten2.data[6] + Ten1.data[58] * Ten2.data[7] + Ten1.data[67] * Ten2.data[8] + Ten1.data[76] * Ten2.data[9],
    Ten1.data[5] * Ten2.data[1] + Ten1.data[14] * Ten2.data[2] + Ten1.data[23] * Ten2.data[3] + Ten1.data[32] * Ten2.data[4] + Ten1.data[41] * Ten2.data[5] + Ten1.data[50] * Ten2.data[6] + Ten1.data[59] * Ten2.data[7] + Ten1.data[68] * Ten2.data[8] + Ten1.data[77] * Ten2.data[9],
    Ten1.data[6] * Ten2.data[1] + Ten1.data[15] * Ten2.data[2] + Ten1.data[24] * Ten2.data[3] + Ten1.data[33] * Ten2.data[4] + Ten1.data[42] * Ten2.data[5] + Ten1.data[51] * Ten2.data[6] + Ten1.data[60] * Ten2.data[7] + Ten1.data[69] * Ten2.data[8] + Ten1.data[78] * Ten2.data[9],
    Ten1.data[7] * Ten2.data[1] + Ten1.data[16] * Ten2.data[2] + Ten1.data[25] * Ten2.data[3] + Ten1.data[34] * Ten2.data[4] + Ten1.data[43] * Ten2.data[5] + Ten1.data[52] * Ten2.data[6] + Ten1.data[61] * Ten2.data[7] + Ten1.data[70] * Ten2.data[8] + Ten1.data[79] * Ten2.data[9],
    Ten1.data[8] * Ten2.data[1] + Ten1.data[17] * Ten2.data[2] + Ten1.data[26] * Ten2.data[3] + Ten1.data[35] * Ten2.data[4] + Ten1.data[44] * Ten2.data[5] + Ten1.data[53] * Ten2.data[6] + Ten1.data[62] * Ten2.data[7] + Ten1.data[71] * Ten2.data[8] + Ten1.data[80] * Ten2.data[9],
    Ten1.data[9] * Ten2.data[1] + Ten1.data[18] * Ten2.data[2] + Ten1.data[27] * Ten2.data[3] + Ten1.data[36] * Ten2.data[4] + Ten1.data[45] * Ten2.data[5] + Ten1.data[54] * Ten2.data[6] + Ten1.data[63] * Ten2.data[7] + Ten1.data[72] * Ten2.data[8] + Ten1.data[81] * Ten2.data[9])
end


"""
  Outer_12_34(Matrix1::SMatrix, Matrix2::SMatrix)::SMatrix

Computes the **outer product** of two second-order tensors (matrices), returning a fourth-order tensor 
represented in a `dim² x dim²` matrix form (Voigt or flattened index notation) using combined indices.

# Arguments
- `Matrix1::SMatrix{dim, dim}`: A second-order tensor (e.g., a stress, strain, or deformation tensor).
- `Matrix2::SMatrix{dim, dim}`: Another second-order tensor to be used in the outer product.

# Returns
- `SMatrix{dim^2, dim^2}`: A statically-sized matrix representing the fourth-order tensor formed from the outer product.
"""
@inline function Outer_12_34(Matrix1::SMatrix, Matrix2::SMatrix)
   dim   =  size(Matrix1,1)
   Out = zeros(Float64,dim*dim, dim*dim)
   for j in 1:dim
       for i in 1:dim
           for l in 1:dim
               for k in 1:dim
                   Out[i + dim * (j - 1), k + dim * (l - 1)] += Matrix1[i,j] * Matrix2[k,l]
               end
           end
       end
   end
   return SMatrix{dim*dim, dim*dim}(Out)
end


"""
  Outer_13_24(Matrix1::SMatrix, Matrix2::SMatrix)::SMatrix

Computes the **outer product** of two second-order tensors (matrices), returning a fourth-order tensor 
represented in a `dim² x dim²` matrix form (Voigt or flattened index notation) using combined indices.

# Arguments
- `Matrix1::SMatrix{dim, dim}`: A second-order tensor (e.g., a stress, strain, or deformation tensor).
- `Matrix2::SMatrix{dim, dim}`: Another second-order tensor to be used in the outer product.

# Returns
- `SMatrix{dim^2, dim^2}`: A statically-sized matrix representing the fourth-order tensor formed from the outer product.
"""
@inline function Outer_13_24(Matrix1::SMatrix, Matrix2::SMatrix)
   dim   =  size(Matrix1,1)
   Out = zeros(Float64,dim*dim, dim*dim)
   for j in 1:dim
       for i in 1:dim
           for l in 1:dim
               for k in 1:dim
                   Out[i + dim * (j - 1), k + dim * (l - 1)] += Matrix1[i,k] * Matrix2[j,l]
               end
           end
       end
   end
   return SMatrix{dim*dim, dim*dim}(Out)
end


"""
  Outer_14_23(Matrix1::SMatrix, Matrix2::SMatrix)::SMatrix

Computes the **outer product** of two second-order tensors (matrices), returning a fourth-order tensor 
represented in a `dim² x dim²` matrix form (Voigt or flattened index notation) using combined indices.

# Arguments
- `Matrix1::SMatrix{dim, dim}`: A second-order tensor (e.g., a stress, strain, or deformation tensor).
- `Matrix2::SMatrix{dim, dim}`: Another second-order tensor to be used in the outer product.

# Returns
- `SMatrix{dim^2, dim^2}`: A statically-sized matrix representing the fourth-order tensor formed from the outer product.
"""
@inline function Outer_14_23(Matrix1::SMatrix, Matrix2::SMatrix)
   dim   =  size(Matrix1,1)
   Out = zeros(Float64,dim*dim, dim*dim)
   for j in 1:dim
       for i in 1:dim
           for l in 1:dim
               for k in 1:dim
                   Out[i + dim * (j - 1), k + dim * (l - 1)] += Matrix1[i,l] * Matrix2[j,k]
               end
           end
       end
   end
   return SMatrix{dim*dim, dim*dim}(Out)
end


"""
  Contraction_IP_JPKL(Matrix1::SMatrix, Matrix2::SMatrix)::SMatrix

Performs a specific tensor contraction between a second-order tensor `Matrix1` (of size `dim x dim`)
and a fourth-order tensor `Matrix2` (represented as a `dim² x dim²` matrix in Voigt or flattened index notation),
returning the result as a new fourth-order tensor (in the same flattened form).

The contraction follows the **index contraction pattern**, where addition is performed for repeated indices.

# Arguments
- `Matrix1::SMatrix{dim, dim}`: A second-order tensor (e.g., a stress or deformation tensor).
- `Matrix2::SMatrix{dim^2, dim^2}`: A fourth-order tensor written in matrix form using combined indices.

# Returns
- `SMatrix{dim^2, dim^2}`: The resulting fourth-order tensor in the same matrix representation.
"""
@inline function Contraction_IP_JPKL(Matrix1::SMatrix, Matrix2::SMatrix)
   dim   =  size(Matrix1,1)
   Out   =  zeros(Float64,dim*dim, dim*dim)
   for i in 1:dim
       for j in 1:dim
           for k in 1:dim
               for l in 1:dim
                   for p in 1:dim
                       Out[i + dim * (j - 1), k + dim * (l - 1)] += Matrix1[i,p] * Matrix2[j + dim * (p - 1),k + dim * (l - 1)]
                   end
               end
           end
       end
   end
   return SMatrix{dim*dim, dim*dim}(Out)
end


"""
  Contraction_IP_PJKL(Matrix1::SMatrix, Matrix2::SMatrix)::SMatrix

Performs a specific tensor contraction between a second-order tensor `Matrix1` (of size `dim x dim`)
and a fourth-order tensor `Matrix2` (represented as a `dim² x dim²` matrix in Voigt or flattened index notation),
returning the result as a new fourth-order tensor (in the same flattened form).

The contraction follows the **index contraction pattern**, where addition is performed for repeated indices.

# Arguments
- `Matrix1::SMatrix{dim, dim}`: A second-order tensor (e.g., a stress or deformation tensor).
- `Matrix2::SMatrix{dim^2, dim^2}`: A fourth-order tensor written in matrix form using combined indices.

# Returns
- `SMatrix{dim^2, dim^2}`: The resulting fourth-order tensor in the same matrix representation.
"""
@inline function Contraction_IP_PJKL(Matrix1::SMatrix, Matrix2::SMatrix)
   dim   =  size(Matrix1,1)
   Out   =  zeros(Float64,dim*dim, dim*dim)
   for i in 1:dim
       for j in 1:dim
           for k in 1:dim
               for l in 1:dim
                   for p in 1:dim
                       Out[i + dim * (j - 1), k + dim * (l - 1)] += Matrix1[i,p] * Matrix2[p + dim * (j - 1),k + dim * (l - 1)]
                   end
               end
           end
       end
   end
   return SMatrix{dim*dim, dim*dim}(Out)
end


end