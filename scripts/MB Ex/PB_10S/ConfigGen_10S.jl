using CSV
using DataFrames

conf = []
for x1 in 0:2, x2 in 0:2, x3 in 0:2, x4 in 0:2, x5 in 0:2, x6 in 0:2, x7 in 0:2, x8 in 0:2, x9 in 0:2, x10 in 0:2
  push!(conf,[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
end


r_set = [rand(conf) for i in 1:60]
sort!(r_set)

df = DataFrame(r_set, :auto)
CSV.write("data/csv/Config_N60_EM_PB_10S.csv", df)

# CSV.File("data/csv/Config_N600_EM_PB_10S.csv") |> Tables.Columns