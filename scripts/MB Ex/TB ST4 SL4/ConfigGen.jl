using CSV
using DataFrames

# ------------- Required configurations -------------------
conf_a = [Int.(digits(i-1,base=2,pad=4)) for i in 1:16]
conf_a = reduce(hcat,conf_a)
conf_ = vcat(conf_a,zeros(Int8, (12,16)))
conf_ = hcat(conf_,vcat(zeros(Int8,4,16),conf_a,zeros(Int8,8,16)))
conf_ = hcat(conf_,vcat(zeros(Int8,8,16),conf_a,zeros(Int8,4,16)))
conf_ = hcat(conf_,vcat(zeros(Int8,12,16),conf_a))
df = DataFrame(conf_, :auto)
CSV.write("data/csv/EM_TB_ST4_SL4_Conf0.csv", df)

conf = []
for i1 in 0:1, i2 in 0:1, i3 in 0:1, i4 in 0:1, i5 in 0:1, i6 in 0:1, i7 in 0:1, i8 in 0:1, i9 in 0:1, i10 in 0:1, i11 in 0:1, i12 in 0:1, i13 in 0:1, i14 in 0:1, i15 in 0:1, i16 in 0:1
    push!(conf,[i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16])
end
conf = reduce(hcat,conf)
r_set = [rand(conf) for i in 1:1000]
r_set = reduce(hcat,r_set)
df = DataFrame(r_set, :auto)
CSV.write("data/csv/EM_TB_ST4_SL4_ConfRand.csv", df)