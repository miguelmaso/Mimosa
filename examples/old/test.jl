using Gridap
using Gridap.TensorValues

gradu = TensorValue(9,9,Tuple(1:81))

gradp = VectorValue(0.8, 0.7)

Ten1[1] * Ten2[1] + Ten1[10] * Ten2[2] + Ten1[19] * Ten2[3] + Ten1[28] * Ten2[4] + Ten1[37] * Ten2[5] + Ten1[46] * Ten2[6] + Ten1[55] * Ten2[7] + Ten1[64] * Ten2[8] + Ten1[73] * Ten2[9],
    Ten1[2] * Ten2[1] + Ten1[11] * Ten2[2] + Ten1[20] * Ten2[3] + Ten1[29] * Ten2[4] + Ten1[38] * Ten2[5] + Ten1[47] * Ten2[6] + Ten1[56] * Ten2[7] + Ten1[65] * Ten2[8] + Ten1[74] * Ten2[9],
    Ten1[3] * Ten2[1] + Ten1[12] * Ten2[2] + Ten1[21] * Ten2[3] + Ten1[30] * Ten2[4] + Ten1[39] * Ten2[5] + Ten1[48] * Ten2[6] + Ten1[57] * Ten2[7] + Ten1[66] * Ten2[8] + Ten1[75] * Ten2[9],
    Ten1[4] * Ten2[1] + Ten1[13] * Ten2[2] + Ten1[22] * Ten2[3] + Ten1[31] * Ten2[4] + Ten1[40] * Ten2[5] + Ten1[49] * Ten2[6] + Ten1[58] * Ten2[7] + Ten1[67] * Ten2[8] + Ten1[76] * Ten2[9],
    Ten1[5] * Ten2[1] + Ten1[14] * Ten2[2] + Ten1[23] * Ten2[3] + Ten1[32] * Ten2[4] + Ten1[41] * Ten2[5] + Ten1[50] * Ten2[6] + Ten1[59] * Ten2[7] + Ten1[68] * Ten2[8] + Ten1[77] * Ten2[9],
    Ten1[6] * Ten2[1] + Ten1[15] * Ten2[2] + Ten1[24] * Ten2[3] + Ten1[33] * Ten2[4] + Ten1[42] * Ten2[5] + Ten1[51] * Ten2[6] + Ten1[60] * Ten2[7] + Ten1[69] * Ten2[8] + Ten1[78] * Ten2[9],
    Ten1[7] * Ten2[1] + Ten1[16] * Ten2[2] + Ten1[25] * Ten2[3] + Ten1[34] * Ten2[4] + Ten1[43] * Ten2[5] + Ten1[52] * Ten2[6] + Ten1[61] * Ten2[7] + Ten1[70] * Ten2[8] + Ten1[79] * Ten2[9],
    Ten1[8] * Ten2[1] + Ten1[17] * Ten2[2] + Ten1[26] * Ten2[3] + Ten1[35] * Ten2[4] + Ten1[44] * Ten2[5] + Ten1[53] * Ten2[6] + Ten1[62] * Ten2[7] + Ten1[71] * Ten2[8] + Ten1[80] * Ten2[9],
    Ten1[9] * Ten2[1] + Ten1[18] * Ten2[2] + Ten1[27] * Ten2[3] + Ten1[36] * Ten2[4] + Ten1[45] * Ten2[5] + Ten1[54] * Ten2[6] + Ten1[63] * Ten2[7] + Ten1[72] * Ten2[8] + Ten1[81] * Ten2[9]
