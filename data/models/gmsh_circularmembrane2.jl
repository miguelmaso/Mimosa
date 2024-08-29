using Gmsh: Gmsh, gmsh

function Arc_Sec(R,r,θ_st,θ,ct_pt,nr,ntan,lc)
    geo = gmsh.model.geo
    #print([nl,nw,nt])
    p = []
    append!(p,geo.addPoint(ct_pt[1] + r*cos(θ_st), ct_pt[2] + r*sin(θ_st), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + R*cos(θ_st), ct_pt[2] + R*sin(θ_st), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + R*cos(θ_st + θ), ct_pt[2] + R*sin(θ_st + θ), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + r*cos(θ_st + θ),ct_pt[2] + r*sin(θ_st + θ),ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1],ct_pt[2],ct_pt[3], lc, -1))
    l = []
    append!(l,geo.addLine(p[1],p[2],-1))
    append!(l,geo.addCircleArc(p[2],p[5],p[3],-1))
    append!(l,geo.addLine(p[3],p[4],-1))
    append!(l,geo.addCircleArc(p[4],p[5],p[1],-1))
    loop = geo.addCurveLoop(l,-1)
    rec = geo.addPlaneSurface([loop],-1)
    geo.mesh.setTransfiniteCurve(l[1], nr )
    geo.mesh.setTransfiniteCurve(l[2], ntan )
    geo.mesh.setTransfiniteCurve(l[3], nr )
    geo.mesh.setTransfiniteCurve(l[4], ntan )
    geo.synchronize()
    return rec
end

function Center_Sec(R,θ_st,θ,ct_pt,nr,ntan,lc)
    geo = gmsh.model.geo
    #print([nl,nw,nt])
    p = []
    append!(p,geo.addPoint(ct_pt[1] + R*cos(θ_st), ct_pt[2] + R*sin(θ_st), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1] + R*cos(θ_st + θ), ct_pt[2] + R*sin(θ_st + θ), ct_pt[3], lc, -1))
    append!(p,geo.addPoint(ct_pt[1],ct_pt[2],ct_pt[3], lc, -1))
    l = []
    append!(l,geo.addCircleArc(p[1],p[3],p[2],-1))
    append!(l,geo.addLine(p[2],p[3],-1))
    append!(l,geo.addLine(p[3],p[1],-1))
    loop = geo.addCurveLoop(l,-1)
    rec = geo.addPlaneSurface([loop],-1)
    geo.mesh.setTransfiniteCurve(l[1], ntan )
    geo.mesh.setTransfiniteCurve(l[2], nr )
    geo.mesh.setTransfiniteCurve(l[3], nr )
    geo.synchronize()
    return rec
end

function run()
    gmsh.initialize()
    ct_pt,nr,ntan,lc = [0.0,0.0,0.0],2,4,2.0e-4
    surf_list = []
    n_sec_t = 4
    n:sec_r = 4
    Δθ = (2*pi)/n_sec_t
    R_ = 25.0e-3
    ΔR = R_/n_sec_r
    r = 0.0
    R = ΔR
    for i in 1:n_sec_r
        θ_st = 0.0
        for j in 1:n_sec_t
            if i == 1
                push!(surf_list,Center_Sec(R,θ_st,Δθ,ct_pt,nr,ntan,lc))
            else
                push!(surf_list,Arc_Sec(R,r,θ_st,Δθ,ct_pt,nr,ntan,lc))
            end
            θ_st += Δθ
        end
        r = R
        R += ΔR
    end
    if !("-nopopup" in ARGS)
        gmsh.fltk.run()
    end
    # output_file = joinpath(dirname(@__FILE__), model_name*".msh")
    # gmsh.write(output_file)
    Gmsh.finalize() 
end