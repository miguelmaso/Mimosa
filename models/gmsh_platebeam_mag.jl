using Gmsh: Gmsh, gmsh
Gmsh.initialize()

gmsh.model.add("mesh_platebeam_mag")

# parameters
const L=40;      # beam length
const W=8;       # beam width
const T=0.4;     # beam thickness

# const nl=40; # X element size
# const nw=10; # Y element size
# const nt=4; # Z element size


const nl=10; # X element size
const nw=5; # Y element size
const nt=2; # Z element size


const lc = 1.0; # characteristic length for meshing
 
geo = gmsh.model.geo
#  fixed section at (0,0,0)
geo.addPoint(0, 0, 0, lc, 1)
geo.addPoint(L, 0, 0, lc, 2)
geo.addPoint(0, 0, T, lc, 3)
geo.addPoint(0, W, 0, lc, 4)
geo.addPoint(0, W, T, lc, 5)
geo.addPoint(L, W, 0, lc, 6)
geo.addPoint(L, 0, T, lc, 7)
geo.addPoint(L, W, T, lc, 8)


geo.addLine(1, 3, 1)
geo.addLine(3, 5, 2)
geo.addLine(5, 4, 3)
geo.addLine(4, 1, 4)
geo.addLine(7, 8, 5)
geo.addLine(8, 6, 6)
geo.addLine(6, 2, 7)
geo.addLine(2, 7, 8)
geo.addLine(3, 7, 9)
geo.addLine(2, 1, 10)
geo.addLine(5, 8, 11)
geo.addLine(6, 4, 12)


geo.addCurveLoop([1,2,3,4], 1)
geo.addCurveLoop([5,6,7,8], 2)
geo.addCurveLoop([1,9,-8,10], 3)
geo.addCurveLoop([3,-12,-6,-11],4)
geo.addCurveLoop([-9,2,11,-5], 5)
geo.addCurveLoop([4,-10,-7,12], 6)

 
for i in [1,3,6,8]
      geo.mesh.setTransfiniteCurve(i, nt )  
end
for i in [2,4,5,7]
      geo.mesh.setTransfiniteCurve(i, nw )  
end
for i in [10,12,9,11]
      geo.mesh.setTransfiniteCurve(i, nl )  
end 

for i in 1:6
      geo.addPlaneSurface([i], i)
      geo.mesh.setTransfiniteSurface(i)
      geo.mesh.setRecombine(2,i)
end  
  
geo.addSurfaceLoop([1,2,3,4,5,6], 1)
geo.addVolume([1], 1)
geo.mesh.setTransfiniteVolume(1)

gmsh.model.geo.extrude([(2, 5)], 0.0, 0.0,T , [nt-1], [1],true )

for i in [20,19,28,24]
      geo.mesh.setTransfiniteCurve(i, nt )  
end 

for i in [15,17]
      geo.mesh.setTransfiniteCurve(i, nw )  
end 

for i in [14,16]
      geo.mesh.setTransfiniteCurve(i, nl )  
end 

for i in [25,21,33,29,34]
      geo.mesh.setTransfiniteSurface(i )  
      geo.mesh.setRecombine(2,i)
end 
 
 geo.mesh.setTransfiniteVolume(2)
 gmsh.model.geo.synchronize()


# # Generate mesh
 gmsh.model.mesh.generate(3) 

 gmsh.model.addPhysicalGroup(3, [1,2], 1, "Volume")  

 gmsh.model.addPhysicalGroup(0, [1,3,10,5,14,4], 1,"fixedend")  
 gmsh.model.addPhysicalGroup(1, [1,3,4,2,15,20,24], 1, "fixedend")  
 gmsh.model.addPhysicalGroup(2, [1, 25], 1, "fixedend")  

output_file = joinpath(dirname(@__FILE__), "mesh_platebeam_mag.msh")
gmsh.write(output_file)

 # Launch the GUI to see the results:
if !("-nopopup" in ARGS)
      #     gmsh.fltk.run()
end

Gmsh.finalize()