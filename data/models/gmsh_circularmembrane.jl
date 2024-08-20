using Gmsh: Gmsh, gmsh

function generateRing(D,d,L,n_sec_t,ct_pt,nl,nw,nt,ntan,lc,n_sec,model_name)
    R = D/2
    r = d/2
    geo = gmsh.model.geo
    #print([nl,nw,nt])
    p = []
    append!(p,geo.addPoint(ct_pt[1], ct_pt[2]+r, ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1], ct_pt[2]+R, ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1], ct_pt[2]+R, ct_pt[3]+L, lc, -1))
    append!(p,geo.addPoint(ct_pt[1],ct_pt[2]+r,ct_pt[3]+L, lc, -1))
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

    rec = (0,rec)
    for i in 1:n_sec_t
        prism1 = geo.revolve(([2,rec[2]]), ct_pt[1], ct_pt[2], ct_pt[3], 0, 0, 1, pi/(n_sec_t/2), [ntan], [], true)
        rec = prism1[1]
    end
    
    geo.synchronize()
end

function physicalG(L,D)
    surfaces = gmsh.model.getEntities(2)
    i = 1
    j = 1
    k = 1
    l = 1
    bottom_surf = []
    mid_surf = []
    top_surf = []
    for surface in surfaces
        lines = gmsh.model.getBoundary(surface)
        lines_list = []
        point_list = []
        Z_list = []
        for line in lines
            append!(lines_list,abs(line[2]))
            points = gmsh.model.getBoundary(line)
            for point in points
                pt1 = gmsh.model.getValue(point[1],point[2],[])
                if pt1[1] == 0.0 && pt1[2] == 0.0 && pt1[3] == 0.0
                    gmsh.model.addPhysicalGroup(point[1], [point[2]], -1,"point_xyz")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_xyz")
                    gmsh.model.addPhysicalGroup(point[1]+2, [], -1,"point_xyz")
                elseif pt1[1] == 0.0 && pt1[2] == 0.0 && pt1[3] == 2*L
                    gmsh.model.addPhysicalGroup(point[1], [point[2]], -1,"point_xy")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_xy")
                    gmsh.model.addPhysicalGroup(point[1]+2, [], -1,"point_xy")
                elseif pt1[1] == 0.0 &&  (1-1e-3)*(D/2)<pt1[2] && pt1[2]<(D/2)*(1+1e-3)&& pt1[3] == 0.0
                    gmsh.model.addPhysicalGroup(point[1], [point[2]], -1,"point_x")
                    gmsh.model.addPhysicalGroup(point[1]+1, [], -1,"point_x")
                    gmsh.model.addPhysicalGroup(point[1]+2, [], -1,"point_x")
                end 
                append!(point_list,point[2])
                push!(Z_list,pt1[3])
            end
        end
        Z_avg = sum(Z_list)/length(Z_list)
        if Z_avg==0.0
            gmsh.model.addPhysicalGroup(0, point_list, -1,"bottom_surf_$i")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "bottom_surf_$i")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "bottom_surf_$i")
            push!(bottom_surf,"bottom_surf_$i")
            i = i + 1
        elseif (1-1e-3)*L<Z_avg && Z_avg<L*(1+1e-3)
            gmsh.model.addPhysicalGroup(0, point_list, -1,"mid_surf_$j")
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "mid_surf_$j")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "mid_surf_$j")
            push!(mid_surf,"mid_surf_$j")
            j = j + 1
        elseif (1-1e-3)*L*2<Z_avg && Z_avg<L*(1+1e-3)*2
            gmsh.model.addPhysicalGroup(0, point_list, -1,"top_surf_$k")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "top_surf_$k")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "top_surf_$k")
            push!(top_surf,"top_surf_$k")
            k = k + 1
        else
            gmsh.model.addPhysicalGroup(0, point_list, -1,"surf_$l")  
            gmsh.model.addPhysicalGroup(1, lines_list, -1, "surf_$l")  
            gmsh.model.addPhysicalGroup(2, [surface[2]], -1, "surf_$l")
            l = l + 1
        end
    end
    
    vol = gmsh.model.getEntities(3)

    vol_list = []
    for v in vol
        append!(vol_list,v[2])
    end
    gmsh.model.addPhysicalGroup(3, vol_list, -1, "Volume_")
    return     bottom_surf, mid_surf, top_surf
end

function run()
    model_name = "CircularMambrane"
    D_ini, d, L, n_sec_t = 25.0e-3, 0.0, 0.4e-3, 4
    # ct_pt, nl, nw, nt, ntan, lc = [0.0,0.0,0.0], 1, 2, 1, 2, 2.0e-4 
    #=
    This configuration worked well to enforce only hex elements, then 1 refinement ensured better mesh size.
    However, it took too long for a simulation to finish. So, for the first examples the fallowing configuration with 1 
    refinment was used.
    =#
    ct_pt, nl, nw, nt, ntan, lc = [0.0,0.0,0.0], 1, 1, 1, 1, 2.0e-4
    #=
    There was no way to use ntan higher than 2 while revolving
    The error was related to an extruded vertex that should have been at the center in thi initial ring 
    since that ring has no hole in the center  and the elements had to be changed to a prism1
    futher development may be required, but at time no more time was spent into researching this issue.
    Instead some refiment of the mesh was achieved by increasin the n_sec_t to 8 instead of 4, each with 
    a ntan of 2; this is equivalent to using a ntan of 4. However, this affects the number of control surfaces
    this had to be considered when implementing the bc, such that it was the equivalent of having surface for 
    each 2 coalecent ones
    =#
    n_sec, model_name = nothing, model_name
    gmsh.initialize()
    D = D_ini
    for i in 1:4
        # ntan = ntan + i
        ct_pt[3] = 0.0
        generateRing(D,d,L,n_sec_t,ct_pt,nl,nw,nt,ntan,lc,n_sec,model_name)
        ct_pt[3] += L
        generateRing(D,d,L,n_sec_t,ct_pt,nl,nw,nt,ntan,lc,n_sec,model_name)
        d = D
        D += D_ini
    end
    bottom_surf, mid_surf, top_surf = physicalG(L,D-D_ini)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.refine()
    println(bottom_surf)
    println(mid_surf)
    println(top_surf)

    if !("-nopopup" in ARGS)
        gmsh.fltk.run()
    end
    output_file = joinpath(dirname(@__FILE__), model_name*".msh")
    gmsh.write(output_file)
    Gmsh.finalize() 
end