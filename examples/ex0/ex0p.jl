# using Gridap
# using GridapGmsh
# using GridapPETSc
# using GridapDistributed
# using PartitionedArrays
# function main(ranks)
#   options = "-ksp_type cg -pc_type gamg -ksp_monitor"
#   GridapPETSc.with(args=split(options)) do
#   mesh_file = joinpath(dirname(@__FILE__), "demo.msh")
#   model = GmshDiscreteModel(ranks,mesh_file) 
#   writevtk(model, "results/model")
#     order = 1
#     dirichlet_tags = ["boundary1","boundary2"]
#     u_boundary1(x) = 0.0
#     u_boundary2(x) = 1.0
#     reffe = ReferenceFE(lagrangian,Float64,order)
#     V = TestFESpace(model,reffe,dirichlet_tags=dirichlet_tags)
#     U = TrialFESpace(V,[u_boundary1,u_boundary2])
#     Ω = Interior(model)
#     dΩ = Measure(Ω,2*order)
#     a(u,v) = ∫( ∇(u)⋅∇(v) )dΩ
#     l(v) = 0
#     op = AffineFEOperator(a,l,U,V)
#     solver = PETScLinearSolver()
#     uh = solve(solver,op)
#     writevtk(Ω,"results/demo",cellfields=["uh"=>uh])
#   end
# end
# with_mpi() do distribute 
#   ranks = distribute_with_mpi(LinearIndices((1,)))
#    main(ranks)
# end


using Gridap
using GridapDistributed
using PartitionedArrays
using MPI
partition = (2,2)
prun(mpi, partition) do parts
#   domain = (0,1,0,1)
#   mesh_partition = (4,4)
#   model = CartesianDiscreteModel(parts, domain, mesh_partition)
#   order = 2
#   u((x,y)) = (x+y)^order
#   f(x) = -Δ(u, x)
#   reffe = ReferenceFE(lagrangian, Float64, order)
#   V = TestFESpace(model, reffe, dirichlet_tags="boundary")
#   U = TrialFESpace(u, V)
#   Ω = Triangulation(model)
#   dΩ = Measure(Ω, 2*order)
#   a(u, v) = ∫(∇(v)⋅∇(u))dΩ
#   l(v) = ∫(v*f)dΩ
#   op = AffineFEOperator(a, l, U, V)
#   uh = solve(op)
#   writevtk(Ω, "results", cellfields=["uh"=>uh, "grad_uh"=>∇(uh)])
end

# julia --project=. -O3 --check-bounds=no --color=yes -J ./test/compile/Mimosa.dylib ./examples/ex0/ex0p.jl $1
# mpirun -n 4 julia --project=. example.jl.