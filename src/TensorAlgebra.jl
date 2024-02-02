module TensorAlgebra

using Gridap
using Gridap.TensorValues
import Base: *

export (*)
export cross_I4


function Gridap.TensorValues.cross(A::TensorValue{3,3,Float64}, C::TensorValue{9,9,Float64})
  TensorValue(A.data[9] * C.data[37] - A.data[8] * C.data[46] - A.data[6] * C.data[64] + A.data[5] * C.data[73],
    A.data[9] * C.data[38] - A.data[8] * C.data[47] - A.data[6] * C.data[65] + A.data[5] * C.data[74],
    A.data[9] * C.data[39] - A.data[8] * C.data[48] - A.data[6] * C.data[66] + A.data[5] * C.data[75],
    A.data[9] * C.data[40] - A.data[8] * C.data[49] - A.data[6] * C.data[67] + A.data[5] * C.data[76],
    A.data[9] * C.data[41] - A.data[8] * C.data[50] - A.data[6] * C.data[68] + A.data[5] * C.data[77],
    A.data[9] * C.data[42] - A.data[8] * C.data[51] - A.data[6] * C.data[69] + A.data[5] * C.data[78],
    A.data[9] * C.data[43] - A.data[8] * C.data[52] - A.data[6] * C.data[70] + A.data[5] * C.data[79],
    A.data[9] * C.data[44] - A.data[8] * C.data[53] - A.data[6] * C.data[71] + A.data[5] * C.data[80],
    A.data[9] * C.data[45] - A.data[8] * C.data[54] - A.data[6] * C.data[72] + A.data[5] * C.data[81],
    A.data[7] * C.data[46] - A.data[9] * C.data[28] + A.data[6] * C.data[55] - A.data[4] * C.data[73],
    A.data[7] * C.data[47] - A.data[9] * C.data[29] + A.data[6] * C.data[56] - A.data[4] * C.data[74],
    A.data[7] * C.data[48] - A.data[9] * C.data[30] + A.data[6] * C.data[57] - A.data[4] * C.data[75],
    A.data[7] * C.data[49] - A.data[9] * C.data[31] + A.data[6] * C.data[58] - A.data[4] * C.data[76],
    A.data[7] * C.data[50] - A.data[9] * C.data[32] + A.data[6] * C.data[59] - A.data[4] * C.data[77],
    A.data[7] * C.data[51] - A.data[9] * C.data[33] + A.data[6] * C.data[60] - A.data[4] * C.data[78],
    A.data[7] * C.data[52] - A.data[9] * C.data[34] + A.data[6] * C.data[61] - A.data[4] * C.data[79],
    A.data[7] * C.data[53] - A.data[9] * C.data[35] + A.data[6] * C.data[62] - A.data[4] * C.data[80],
    A.data[7] * C.data[54] - A.data[9] * C.data[36] + A.data[6] * C.data[63] - A.data[4] * C.data[81],
    A.data[8] * C.data[28] - A.data[7] * C.data[37] - A.data[5] * C.data[55] + A.data[4] * C.data[64],
    A.data[8] * C.data[29] - A.data[7] * C.data[38] - A.data[5] * C.data[56] + A.data[4] * C.data[65],
    A.data[8] * C.data[30] - A.data[7] * C.data[39] - A.data[5] * C.data[57] + A.data[4] * C.data[66],
    A.data[8] * C.data[31] - A.data[7] * C.data[40] - A.data[5] * C.data[58] + A.data[4] * C.data[67],
    A.data[8] * C.data[32] - A.data[7] * C.data[41] - A.data[5] * C.data[59] + A.data[4] * C.data[68],
    A.data[8] * C.data[33] - A.data[7] * C.data[42] - A.data[5] * C.data[60] + A.data[4] * C.data[69],
    A.data[8] * C.data[34] - A.data[7] * C.data[43] - A.data[5] * C.data[61] + A.data[4] * C.data[70],
    A.data[8] * C.data[35] - A.data[7] * C.data[44] - A.data[5] * C.data[62] + A.data[4] * C.data[71],
    A.data[8] * C.data[36] - A.data[7] * C.data[45] - A.data[5] * C.data[63] + A.data[4] * C.data[72],
    A.data[8] * C.data[19] - A.data[9] * C.data[10] + A.data[3] * C.data[64] - A.data[2] * C.data[73],
    A.data[8] * C.data[20] - A.data[9] * C.data[11] + A.data[3] * C.data[65] - A.data[2] * C.data[74],
    A.data[8] * C.data[21] - A.data[9] * C.data[12] + A.data[3] * C.data[66] - A.data[2] * C.data[75],
    A.data[8] * C.data[22] - A.data[9] * C.data[13] + A.data[3] * C.data[67] - A.data[2] * C.data[76],
    A.data[8] * C.data[23] - A.data[9] * C.data[14] + A.data[3] * C.data[68] - A.data[2] * C.data[77],
    A.data[8] * C.data[24] - A.data[9] * C.data[15] + A.data[3] * C.data[69] - A.data[2] * C.data[78],
    A.data[8] * C.data[25] - A.data[9] * C.data[16] + A.data[3] * C.data[70] - A.data[2] * C.data[79],
    A.data[8] * C.data[26] - A.data[9] * C.data[17] + A.data[3] * C.data[71] - A.data[2] * C.data[80],
    A.data[8] * C.data[27] - A.data[9] * C.data[18] + A.data[3] * C.data[72] - A.data[2] * C.data[81],
    A.data[9] * C.data[1] - A.data[7] * C.data[19] - A.data[3] * C.data[55] + A.data[1] * C.data[73],
    A.data[9] * C.data[2] - A.data[7] * C.data[20] - A.data[3] * C.data[56] + A.data[1] * C.data[74],
    A.data[9] * C.data[3] - A.data[7] * C.data[21] - A.data[3] * C.data[57] + A.data[1] * C.data[75],
    A.data[9] * C.data[4] - A.data[7] * C.data[22] - A.data[3] * C.data[58] + A.data[1] * C.data[76],
    A.data[9] * C.data[5] - A.data[7] * C.data[23] - A.data[3] * C.data[59] + A.data[1] * C.data[77],
    A.data[9] * C.data[6] - A.data[7] * C.data[24] - A.data[3] * C.data[60] + A.data[1] * C.data[78],
    A.data[9] * C.data[7] - A.data[7] * C.data[25] - A.data[3] * C.data[61] + A.data[1] * C.data[79],
    A.data[9] * C.data[8] - A.data[7] * C.data[26] - A.data[3] * C.data[62] + A.data[1] * C.data[80],
    A.data[9] * C.data[9] - A.data[7] * C.data[27] - A.data[3] * C.data[63] + A.data[1] * C.data[81],
    A.data[7] * C.data[10] - A.data[8] * C.data[1] + A.data[2] * C.data[55] - A.data[1] * C.data[64],
    A.data[7] * C.data[11] - A.data[8] * C.data[2] + A.data[2] * C.data[56] - A.data[1] * C.data[65],
    A.data[7] * C.data[12] - A.data[8] * C.data[3] + A.data[2] * C.data[57] - A.data[1] * C.data[66],
    A.data[7] * C.data[13] - A.data[8] * C.data[4] + A.data[2] * C.data[58] - A.data[1] * C.data[67],
    A.data[7] * C.data[14] - A.data[8] * C.data[5] + A.data[2] * C.data[59] - A.data[1] * C.data[68],
    A.data[7] * C.data[15] - A.data[8] * C.data[6] + A.data[2] * C.data[60] - A.data[1] * C.data[69],
    A.data[7] * C.data[16] - A.data[8] * C.data[7] + A.data[2] * C.data[61] - A.data[1] * C.data[70],
    A.data[7] * C.data[17] - A.data[8] * C.data[8] + A.data[2] * C.data[62] - A.data[1] * C.data[71],
    A.data[7] * C.data[18] - A.data[8] * C.data[9] + A.data[2] * C.data[63] - A.data[1] * C.data[72],
    A.data[6] * C.data[10] - A.data[5] * C.data[19] - A.data[3] * C.data[37] + A.data[2] * C.data[46],
    A.data[6] * C.data[11] - A.data[5] * C.data[20] - A.data[3] * C.data[38] + A.data[2] * C.data[47],
    A.data[6] * C.data[12] - A.data[5] * C.data[21] - A.data[3] * C.data[39] + A.data[2] * C.data[48],
    A.data[6] * C.data[13] - A.data[5] * C.data[22] - A.data[3] * C.data[40] + A.data[2] * C.data[49],
    A.data[6] * C.data[14] - A.data[5] * C.data[23] - A.data[3] * C.data[41] + A.data[2] * C.data[50],
    A.data[6] * C.data[15] - A.data[5] * C.data[24] - A.data[3] * C.data[42] + A.data[2] * C.data[51],
    A.data[6] * C.data[16] - A.data[5] * C.data[25] - A.data[3] * C.data[43] + A.data[2] * C.data[52],
    A.data[6] * C.data[17] - A.data[5] * C.data[26] - A.data[3] * C.data[44] + A.data[2] * C.data[53],
    A.data[6] * C.data[18] - A.data[5] * C.data[27] - A.data[3] * C.data[45] + A.data[2] * C.data[54],
    A.data[4] * C.data[19] - A.data[6] * C.data[1] + A.data[3] * C.data[28] - A.data[1] * C.data[46],
    A.data[4] * C.data[20] - A.data[6] * C.data[2] + A.data[3] * C.data[29] - A.data[1] * C.data[47],
    A.data[4] * C.data[21] - A.data[6] * C.data[3] + A.data[3] * C.data[30] - A.data[1] * C.data[48],
    A.data[4] * C.data[22] - A.data[6] * C.data[4] + A.data[3] * C.data[31] - A.data[1] * C.data[49],
    A.data[4] * C.data[23] - A.data[6] * C.data[5] + A.data[3] * C.data[32] - A.data[1] * C.data[50],
    A.data[4] * C.data[24] - A.data[6] * C.data[6] + A.data[3] * C.data[33] - A.data[1] * C.data[51],
    A.data[4] * C.data[25] - A.data[6] * C.data[7] + A.data[3] * C.data[34] - A.data[1] * C.data[52],
    A.data[4] * C.data[26] - A.data[6] * C.data[8] + A.data[3] * C.data[35] - A.data[1] * C.data[53],
    A.data[4] * C.data[27] - A.data[6] * C.data[9] + A.data[3] * C.data[36] - A.data[1] * C.data[54],
    A.data[5] * C.data[1] - A.data[4] * C.data[10] - A.data[2] * C.data[28] + A.data[1] * C.data[37],
    A.data[5] * C.data[2] - A.data[4] * C.data[11] - A.data[2] * C.data[29] + A.data[1] * C.data[38],
    A.data[5] * C.data[3] - A.data[4] * C.data[12] - A.data[2] * C.data[30] + A.data[1] * C.data[39],
    A.data[5] * C.data[4] - A.data[4] * C.data[13] - A.data[2] * C.data[31] + A.data[1] * C.data[40],
    A.data[5] * C.data[5] - A.data[4] * C.data[14] - A.data[2] * C.data[32] + A.data[1] * C.data[41],
    A.data[5] * C.data[6] - A.data[4] * C.data[15] - A.data[2] * C.data[33] + A.data[1] * C.data[42],
    A.data[5] * C.data[7] - A.data[4] * C.data[16] - A.data[2] * C.data[34] + A.data[1] * C.data[43],
    A.data[5] * C.data[8] - A.data[4] * C.data[17] - A.data[2] * C.data[35] + A.data[1] * C.data[44],
    A.data[5] * C.data[9] - A.data[4] * C.data[18] - A.data[2] * C.data[36] + A.data[1] * C.data[45])
end




function Gridap.TensorValues.outer(A::TensorValue{3,3,Float64}, B::TensorValue{3,3,Float64})
  TensorValue(A.data[1] * B.data[1],
    A.data[2] * B.data[1],
    A.data[3] * B.data[1],
    A.data[4] * B.data[1],
    A.data[5] * B.data[1],
    A.data[6] * B.data[1],
    A.data[7] * B.data[1],
    A.data[8] * B.data[1],
    A.data[9] * B.data[1],
    A.data[1] * B.data[2],
    A.data[2] * B.data[2],
    A.data[3] * B.data[2],
    A.data[4] * B.data[2],
    A.data[5] * B.data[2],
    A.data[6] * B.data[2],
    A.data[7] * B.data[2],
    A.data[8] * B.data[2],
    A.data[9] * B.data[2],
    A.data[1] * B.data[3],
    A.data[2] * B.data[3],
    A.data[3] * B.data[3],
    A.data[4] * B.data[3],
    A.data[5] * B.data[3],
    A.data[6] * B.data[3],
    A.data[7] * B.data[3],
    A.data[8] * B.data[3],
    A.data[9] * B.data[3],
    A.data[1] * B.data[4],
    A.data[2] * B.data[4],
    A.data[3] * B.data[4],
    A.data[4] * B.data[4],
    A.data[5] * B.data[4],
    A.data[6] * B.data[4],
    A.data[7] * B.data[4],
    A.data[8] * B.data[4],
    A.data[9] * B.data[4],
    A.data[1] * B.data[5],
    A.data[2] * B.data[5],
    A.data[3] * B.data[5],
    A.data[4] * B.data[5],
    A.data[5] * B.data[5],
    A.data[6] * B.data[5],
    A.data[7] * B.data[5],
    A.data[8] * B.data[5],
    A.data[9] * B.data[5],
    A.data[1] * B.data[6],
    A.data[2] * B.data[6],
    A.data[3] * B.data[6],
    A.data[4] * B.data[6],
    A.data[5] * B.data[6],
    A.data[6] * B.data[6],
    A.data[7] * B.data[6],
    A.data[8] * B.data[6],
    A.data[9] * B.data[6],
    A.data[1] * B.data[7],
    A.data[2] * B.data[7],
    A.data[3] * B.data[7],
    A.data[4] * B.data[7],
    A.data[5] * B.data[7],
    A.data[6] * B.data[7],
    A.data[7] * B.data[7],
    A.data[8] * B.data[7],
    A.data[9] * B.data[7],
    A.data[1] * B.data[8],
    A.data[2] * B.data[8],
    A.data[3] * B.data[8],
    A.data[4] * B.data[8],
    A.data[5] * B.data[8],
    A.data[6] * B.data[8],
    A.data[7] * B.data[8],
    A.data[8] * B.data[8],
    A.data[9] * B.data[8],
    A.data[1] * B.data[9],
    A.data[2] * B.data[9],
    A.data[3] * B.data[9],
    A.data[4] * B.data[9],
    A.data[5] * B.data[9],
    A.data[6] * B.data[9],
    A.data[7] * B.data[9],
    A.data[8] * B.data[9],
    A.data[9] * B.data[9])
end

  
function cross_I4(A::TensorValue{3,3,Float64})
  TensorValue(0.0, 0.0, 0.0, 0.0, A.data[9], -A.data[8], 0.0, -A.data[6], A.data[5], 0.0,0.0,0.0,-A.data[9],
  0.0, A.data[7], A.data[6], 0.0, -A.data[4], 0.0,0.0,0.0, A.data[8], -A.data[7], 0.0, -A.data[5], A.data[4], 0.0, 0.0, -A.data[9],
 A.data[8], 0.0, 0.0, 0.0,0.0, A.data[3], -A.data[2], A.data[9], 0.0, -A.data[7], 0.0, 0.0, 0.0, -A.data[3], 0.0,
 A.data[1], -A.data[8], A.data[7], 0.0, 0.0, 0.0, 0.0, A.data[2], -A.data[1], 0.0, 0.0, A.data[6], -A.data[5], 0.0,
-A.data[3], A.data[2], 0.0, 0.0, 0.0, -A.data[6], 0.0, A.data[4], A.data[3], 0.0, -A.data[1],
  0.0, 0.0, 0.0, A.data[5], -A.data[4], 0.0, -A.data[2], A.data[1], 0.0, 0.0, 0.0, 0.0)
end

function Gridap.TensorValues.cross(A::TensorValue{3,3,Float64}, B::TensorValue{3,3,Float64})
  TensorValue(A.data[5] * B.data[9] - A.data[6] * B.data[8] - A.data[8] * B.data[6] + A.data[9] * B.data[5],
    A.data[6] * B.data[7] - A.data[4] * B.data[9] + A.data[7] * B.data[6] - A.data[9] * B.data[4],
    A.data[4] * B.data[8] - A.data[5] * B.data[7] - A.data[7] * B.data[5] + A.data[8] * B.data[4],
    A.data[3] * B.data[8] - A.data[2] * B.data[9] + A.data[8] * B.data[3] - A.data[9] * B.data[2],
    A.data[1] * B.data[9] - A.data[3] * B.data[7] - A.data[7] * B.data[3] + A.data[9] * B.data[1],
    A.data[2] * B.data[7] - A.data[1] * B.data[8] + A.data[7] * B.data[2] - A.data[8] * B.data[1],
    A.data[2] * B.data[6] - A.data[3] * B.data[5] - A.data[5] * B.data[3] + A.data[6] * B.data[2],
    A.data[3] * B.data[4] - A.data[1] * B.data[6] + A.data[4] * B.data[3] - A.data[6] * B.data[1],
    A.data[1] * B.data[5] - A.data[2] * B.data[4] - A.data[4] * B.data[2] + A.data[5] * B.data[1])
end

function Gridap.TensorValues.inner(Ten1::TensorValue{9,9,Float64}, Ten2::TensorValue{3,3,Float64})
  TensorValue(Ten1.data[1] * Ten2.data[1] + Ten1.data[10] * Ten2.data[2] + Ten1.data[19] * Ten2.data[3] + Ten1.data[28] * Ten2.data[4] + Ten1.data[37] * Ten2.data[5] + Ten1.data[46] * Ten2.data[6] + Ten1.data[55] * Ten2.data[7] + Ten1.data[64] * Ten2.data[8] + Ten1.data[73] * Ten2.data[9],
    Ten1.data[2] * Ten2.data[1] + Ten1.data[11] * Ten2.data[2] + Ten1.data[20] * Ten2.data[3] + Ten1.data[29] * Ten2.data[4] + Ten1.data[38] * Ten2.data[5] + Ten1.data[47] * Ten2.data[6] + Ten1.data[56] * Ten2.data[7] + Ten1.data[65] * Ten2.data[8] + Ten1.data[74] * Ten2.data[9],
    Ten1.data[3] * Ten2.data[1] + Ten1.data[12] * Ten2.data[2] + Ten1.data[21] * Ten2.data[3] + Ten1.data[30] * Ten2.data[4] + Ten1.data[39] * Ten2.data[5] + Ten1.data[48] * Ten2.data[6] + Ten1.data[57] * Ten2.data[7] + Ten1.data[66] * Ten2.data[8] + Ten1.data[75] * Ten2.data[9],
    Ten1.data[4] * Ten2.data[1] + Ten1.data[13] * Ten2.data[2] + Ten1.data[22] * Ten2.data[3] + Ten1.data[31] * Ten2.data[4] + Ten1.data[40] * Ten2.data[5] + Ten1.data[49] * Ten2.data[6] + Ten1.data[58] * Ten2.data[7] + Ten1.data[67] * Ten2.data[8] + Ten1.data[76] * Ten2.data[9],
    Ten1.data[5] * Ten2.data[1] + Ten1.data[14] * Ten2.data[2] + Ten1.data[23] * Ten2.data[3] + Ten1.data[32] * Ten2.data[4] + Ten1.data[41] * Ten2.data[5] + Ten1.data[50] * Ten2.data[6] + Ten1.data[59] * Ten2.data[7] + Ten1.data[68] * Ten2.data[8] + Ten1.data[77] * Ten2.data[9],
    Ten1.data[6] * Ten2.data[1] + Ten1.data[15] * Ten2.data[2] + Ten1.data[24] * Ten2.data[3] + Ten1.data[33] * Ten2.data[4] + Ten1.data[42] * Ten2.data[5] + Ten1.data[51] * Ten2.data[6] + Ten1.data[60] * Ten2.data[7] + Ten1.data[69] * Ten2.data[8] + Ten1.data[78] * Ten2.data[9],
    Ten1.data[7] * Ten2.data[1] + Ten1.data[16] * Ten2.data[2] + Ten1.data[25] * Ten2.data[3] + Ten1.data[34] * Ten2.data[4] + Ten1.data[43] * Ten2.data[5] + Ten1.data[52] * Ten2.data[6] + Ten1.data[61] * Ten2.data[7] + Ten1.data[70] * Ten2.data[8] + Ten1.data[79] * Ten2.data[9],
    Ten1.data[8] * Ten2.data[1] + Ten1.data[17] * Ten2.data[2] + Ten1.data[26] * Ten2.data[3] + Ten1.data[35] * Ten2.data[4] + Ten1.data[44] * Ten2.data[5] + Ten1.data[53] * Ten2.data[6] + Ten1.data[62] * Ten2.data[7] + Ten1.data[71] * Ten2.data[8] + Ten1.data[80] * Ten2.data[9],
    Ten1.data[9] * Ten2.data[1] + Ten1.data[18] * Ten2.data[2] + Ten1.data[27] * Ten2.data[3] + Ten1.data[36] * Ten2.data[4] + Ten1.data[45] * Ten2.data[5] + Ten1.data[54] * Ten2.data[6] + Ten1.data[63] * Ten2.data[7] + Ten1.data[72] * Ten2.data[8] + Ten1.data[81] * Ten2.data[9])
end

function Gridap.TensorValues.inner(Ten1::TensorValue{3,9,Float64}, Ten2::TensorValue{3,3,Float64})
  VectorValue(Ten1.data[1] * Ten2.data[1] + Ten1.data[4] * Ten2.data[2] + Ten1.data[7] * Ten2.data[3] + Ten1.data[10] * Ten2.data[4] + Ten1.data[13] * Ten2.data[5] + Ten1.data[16] * Ten2.data[6] + Ten1.data[19] * Ten2.data[7] + Ten1.data[22] * Ten2.data[8] + Ten1.data[25] * Ten2.data[9],
    Ten1.data[2] * Ten2.data[1] + Ten1.data[5] * Ten2.data[2] + Ten1.data[8] * Ten2.data[3] + Ten1.data[11] * Ten2.data[4] + Ten1.data[14] * Ten2.data[5] + Ten1.data[17] * Ten2.data[6] + Ten1.data[20] * Ten2.data[7] + Ten1.data[23] * Ten2.data[8] + Ten1.data[26] * Ten2.data[9],
    Ten1.data[3] * Ten2.data[1] + Ten1.data[6] * Ten2.data[2] + Ten1.data[9] * Ten2.data[3] + Ten1.data[12] * Ten2.data[4] + Ten1.data[15] * Ten2.data[5] + Ten1.data[18] * Ten2.data[6] + Ten1.data[21] * Ten2.data[7] + Ten1.data[24] * Ten2.data[8] + Ten1.data[27] * Ten2.data[9])
end


function Gridap.TensorValues.inner(Ten1::TensorValue{3,9,Float64}, Ten2::VectorValue{3,Float64})
  TensorValue(Ten1.data[1] * Ten2.data[1] + Ten1.data[10] * Ten2.data[2] + Ten1.data[19] * Ten2.data[3],
    Ten1.data[2] * Ten2.data[1] + Ten1.data[11] * Ten2.data[2] + Ten1.data[20] * Ten2.data[3],
    Ten1.data[3] * Ten2.data[1] + Ten1.data[12] * Ten2.data[2] + Ten1.data[21] * Ten2.data[3],
    Ten1.data[4] * Ten2.data[1] + Ten1.data[13] * Ten2.data[2] + Ten1.data[22] * Ten2.data[3],
    Ten1.data[5] * Ten2.data[1] + Ten1.data[14] * Ten2.data[2] + Ten1.data[23] * Ten2.data[3],
    Ten1.data[6] * Ten2.data[1] + Ten1.data[15] * Ten2.data[2] + Ten1.data[24] * Ten2.data[3],
    Ten1.data[7] * Ten2.data[1] + Ten1.data[16] * Ten2.data[2] + Ten1.data[25] * Ten2.data[3],
    Ten1.data[8] * Ten2.data[1] + Ten1.data[17] * Ten2.data[2] + Ten1.data[26] * Ten2.data[3],
    Ten1.data[9] * Ten2.data[1] + Ten1.data[18] * Ten2.data[2] + Ten1.data[27] * Ten2.data[3])
end



@inline function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end

@inline function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end


end