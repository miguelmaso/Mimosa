using Gmsh: Gmsh, gmsh
Gmsh.initialize()

gmsh.model.add("circle1Q")


# parameters
const Ri=1.0;    # interior Radius
const Re=1.5;    # exterior Radius
const t=0.05;     # thickness
const lc = 0.1; # characteristic length for meshing
const divcirc =30;
const divrad  = 5;
const divthick = 4;

geo = gmsh.model.geo
#  fixed section at (0,0,0)
geo.addPoint(Ri, 0, 0, lc, 1)
geo.addPoint(Re, 0, 0, lc, 2)
geo.addPoint(0, Ri, 0, lc, 3)
geo.addPoint(0, Re, 0, lc, 4)
geo.addPoint(0, 0, 0, lc, 5)

geo.addLine(1, 2, 1)
geo.addCircleArc(1, 5, 3, 2)
geo.addLine(3, 4, 3)
geo.addCircleArc(4, 5, 2, 4)
geo.addCurveLoop([1,-4,-3,-2], 1)
 
geo.mesh.setTransfiniteCurve(1, divrad) 
geo.mesh.setTransfiniteCurve(3, divrad) 
geo.mesh.setTransfiniteCurve(2, divcirc) 
geo.mesh.setTransfiniteCurve(4, divcirc) 

geo.addPlaneSurface([1], 1)
geo.mesh.setTransfiniteSurface(1)
geo.mesh.setRecombine(2,1)



gmsh.model.geo.extrude([(2,1)], 0.0, 0.0, t , [divthick-1], [1],true )
geo.mesh.setTransfiniteVolume(1)



gmsh.model.geo.synchronize()



gmsh.model.mesh.generate(3) 

 

output_file = joinpath(dirname(@__FILE__), "circle1Q.msh")
gmsh.write(output_file)

 # Launch the GUI to see the results:
if !("-nopopup" in ARGS)
         gmsh.fltk.run()
end

Gmsh.finalize()