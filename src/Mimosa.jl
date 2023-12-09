module Mimosa
using Gridap.TensorValues
import Base: *


export inner42
export inner32
export logreg
export (*)
export setupfolder

function inner42(Ten1::TensorValue, Ten2::TensorValue)
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

function inner32(Ten1::TensorValue, Ten2::TensorValue)
  VectorValue(Ten1.data[1] * Ten2.data[1] + Ten1.data[4] * Ten2.data[2] + Ten1.data[7] * Ten2.data[3] + Ten1.data[10] * Ten2.data[4] + Ten1.data[13] * Ten2.data[5] + Ten1.data[16] * Ten2.data[6] + Ten1.data[19] * Ten2.data[7] + Ten1.data[22] * Ten2.data[8] + Ten1.data[25] * Ten2.data[9],
    Ten1.data[2] * Ten2.data[1] + Ten1.data[5] * Ten2.data[2] + Ten1.data[8] * Ten2.data[3] + Ten1.data[11] * Ten2.data[4] + Ten1.data[14] * Ten2.data[5] + Ten1.data[17] * Ten2.data[6] + Ten1.data[20] * Ten2.data[7] + Ten1.data[23] * Ten2.data[8] + Ten1.data[26] * Ten2.data[9],
    Ten1.data[3] * Ten2.data[1] + Ten1.data[6] * Ten2.data[2] + Ten1.data[9] * Ten2.data[3] + Ten1.data[12] * Ten2.data[4] + Ten1.data[15] * Ten2.data[5] + Ten1.data[18] * Ten2.data[6] + Ten1.data[21] * Ten2.data[7] + Ten1.data[24] * Ten2.data[8] + Ten1.data[27] * Ten2.data[9])
end

 


@inline function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end

@inline function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end


function logreg(J)
  Jlim=0.01
  if J>= Jlim
    return log(J)
  else
    return log(Jlim)-(3.0/2.0)+(2/Jlim)*J-(1/(2*Jlim^2))*J^2
  end
end


function setupfolder(folder_path::String)
  if !isdir(folder_path)
    mkdir(folder_path)
  else
    rm(folder_path,recursive=true)
    mkdir(folder_path)
  end
end
# Ten1 = TensorValue(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
# 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
# 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
# 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
# 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
# 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
# 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
# 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
# 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)

# Ten2 = TensorValue(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)



end
