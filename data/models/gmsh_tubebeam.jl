using Gmsh: Gmsh, gmsh
gmsh.initialize()

function generateBeamSec(D,d,L,ct_pt,nl,nw,nt,lc,n_sec,model_name)
    R = D/2
    r = d/2
    geo = gmsh.model.geo
    #print([nl,nw,nt])
    p = []
    append!(p,geo.addPoint(ct_pt[1], ct_pt[2]+r, ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1], ct_pt[2]+R, ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1]+L, ct_pt[2]+R, ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1]+L,ct_pt[2]+r,ct_pt[3], lc, -1))
    l = []
    for i in 1:3
        append!(l,geo.addLine(p[i],p[i+1],-1))
    end
    append!(l,geo.addLine(p[4],p[1],-1))
    rec_loop = geo.addCurveLoop(l,-1)
    rec = geo.addPlaneSurface([rec_loop],-1)

    prism1 = geo.revolve(([2,rec]), ct_pt[1], ct_pt[2], ct_pt[3], 1, 0, 0, pi/2)
    rec_ = prism1[1]
    prism2 = geo.revolve((rec_), ct_pt[1], ct_pt[2], ct_pt[3], 1, 0, 0, pi/2)
    rec_ = prism2[1]
    prism3 = geo.revolve((rec_), ct_pt[1], ct_pt[2], ct_pt[3], 1, 0, 0, pi/2)
    rec_ = prism3[1]
    prism4 = geo.revolve((rec_), ct_pt[1], ct_pt[2], ct_pt[3], 1, 0, 0, pi/2)
    
    geo.synchronize()

    surfaces = gmsh.model.getEntities(2)

    i = 1
    for surface in surfaces
        lines = gmsh.model.getBoundary(surface)
        lines_list = []
        point_list = []
        for line in lines
            append!(lines_list,abs(line[2]))
            points = gmsh.model.getBoundary(line)
            for point in points
                append!(point_list,point[2])
            end
        end
        gmsh.model.addPhysicalGroup(0, point_list, i,"surf_$i")  
        gmsh.model.addPhysicalGroup(1, lines_list, i, "surf_$i")  
        gmsh.model.addPhysicalGroup(2, [surface[2]], i, "surf_$i")
        i = i + 1
    end
    
    vol = gmsh.model.getEntities(3)
    vol_list = []
    for v in vol
        append!(vol_list,v[2])
    end
    gmsh.model.addPhysicalGroup(3, vol_list, n_sec, "Volume_$n_sec")

    gmsh.model.mesh.generate(3)
    output_file = joinpath(dirname(@__FILE__), model_name*".msh")
    gmsh.write(output_file)
    if !("-nopopup" in ARGS)
        gmsh.fltk.run()
    end
    Gmsh.finalize()
end

# parameters
L=100e-3;      # beam length
D=10e-3;       # beam width
d=9e-3;     # beam thickness
ct_pt = [0,0,0]
n_sec = 1
# const nl=40; # X element size
# const nw=10; # Y element size
# const nt=4; # Z element size


nl=18; # X element size
nw=8; # Y element size
nt=2; # Z element size

model_name = "TubeBeam"

lc = 2.0e-3; # characteristic length for meshing

prism1 = generateBeamSec(D,d,L,ct_pt,nl,nw,nt,lc,n_sec,model_name)

# function generateBeam(L,W,T,nl,nw,nt,lc,fract,model_name)
#     append!(fract,1 - sum(fract))
#     gmsh.model.add(model_name)
#     n_sec = length(fract)
#     st_pt = [0 0 0]
#     for i in 1:n_sec
#         nsec = i
#         L_sec = L*fract[i]
#         #nl = ceil(L_sec/2)
#         # nw = ceil(W/nw)
#         generateBeamSec(L_sec,W,T,st_pt,nl,nw,nt,lc,nsec)
#         st_pt = [st_pt[1]+L_sec 0 0]
#     end
#     gmsh.model.mesh.generate(3)
#     output_file = joinpath(dirname(@__FILE__), model_name*".msh")
#     gmsh.write(output_file)
#     if !("-nopopup" in ARGS)
#         gmsh.fltk.run()
#     end
#     Gmsh.finalize()
# end

# generateBeam(L,W,T,nl,nw,nt,lc,fract,model_name)