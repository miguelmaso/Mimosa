using Gmsh: Gmsh, gmsh
gmsh.initialize()
function generatePrism(L,W,T,st_pt,nl,nw,nt,lc)
    model_name = "Unit_Cube"
    gmsh.model.add(model_name)
    geo = gmsh.model.geo
    p = []
    append!(p,geo.addPoint(st_pt[1], st_pt[2], st_pt[3], lc, -1))
    append!(p,geo.addPoint(st_pt[1]+L, st_pt[2], st_pt[3], lc, -1))
    append!(p,geo.addPoint(st_pt[1]+L, st_pt[2]+W, st_pt[3], lc, -1))
    append!(p,geo.addPoint(st_pt[1],st_pt[2]+W,st_pt[3], lc, -1))
    l = []
    for i in 1:3
        append!(l,geo.addLine(p[i],p[i+1],-1))
    end
    append!(l,geo.addLine(p[4],p[1],-1))
    rec_loop = geo.addCurveLoop(l,-1)
    rec = geo.addPlaneSurface([rec_loop],-1)
    geo.synchronize()
    lines = gmsh.model.getBoundary([2,rec])
    for line in lines
        pts = gmsh.model.getBoundary(line)
        pt1 = gmsh.model.getValue(pts[1][1],pts[1][2],[])
        pt2 = gmsh.model.getValue(pts[2][1],pts[2][2],[])
        pt_dif = pt2-pt1
        if pt_dif[1]==0.0 && pt_dif[2]==0.0
            geo.mesh.setTransfiniteCurve(line[2], nt )
            C1 = 1
        elseif pt_dif[2]==0.0 && pt_dif[3]==0.0
            geo.mesh.setTransfiniteCurve(line[2], nl )
            C2=2
        else
            geo.mesh.setTransfiniteCurve(line[2], nw )
            c3=3
        end
    end
    surfaces = gmsh.model.getEntities(2)
    for surface in surfaces
        geo.mesh.setTransfiniteSurface(surface[2])
        geo.mesh.setRecombine(2,surface[2])
    end


    prism1 = geo.extrude([(2,rec)],0.0,0.0,T,[nt-1],[1],true)
    geo.synchronize()
    surfaces = gmsh.model.getBoundary([prism1[2][1],prism1[2][2]])
    for surface in surfaces
        surf = abs(surface[2])
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
        gmsh.model.addPhysicalGroup(0, point_list, surf,"face_$surf")  
        gmsh.model.addPhysicalGroup(1, lines_list, surf, "face_$surf")  
        gmsh.model.addPhysicalGroup(2, [surface[2]], surf, "face_$surf")
    end
    gmsh.model.mesh.generate(3)
    output_file = joinpath(dirname(@__FILE__), model_name*".msh")
    gmsh.write(output_file)
    if !("-nopopup" in ARGS)
        gmsh.fltk.run()
    end
    Gmsh.finalize()
end
L,W,T,st_pt,nl,nw,nt,lc = 1, 1, 1, [0,0,0], 10, 10, 10, 1
generatePrism(L,W,T,st_pt,nl,nw,nt,lc)