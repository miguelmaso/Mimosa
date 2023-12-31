using Gmsh: Gmsh, gmsh
Gmsh.initialize()

gmsh.model.add("model_test")

# parameters
const L=1;      # beam length
const W=1;       # beam width
const T=0.5;     # beam thickness

const nl=3; # X element size
const nw=3; # Y element size
const nt=1; # Z element size

const lc = 1.0; # characteristic length for meshing
 
geo = gmsh.model.geo
#  fixed section at (0,0,0)
geo.addPoint(0, 0, 0, lc, 1)
geo.addPoint(0, 0, L, lc, 2)
geo.addPoint(0, T, 0, lc, 3)
geo.addPoint(W, 0, 0, lc, 4)
geo.addPoint(W, T, 0, lc, 5)
geo.addPoint(W, 0, L, lc, 6)
geo.addPoint(0, T, L, lc, 7)
geo.addPoint(W, T, L, lc, 8)


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

gmsh.model.geo.extrude([(2, 5)], 0.0, T,0.0 , [nt], [1],true )

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


# Generate mesh
gmsh.model.mesh.generate(3) 

gmsh.model.addPhysicalGroup(1, [2,5,9,11], 1,"midsuf")  
gmsh.model.addPhysicalGroup(1, [14,15,16,17], 2, "topsuf")  
gmsh.model.addPhysicalGroup(1, [4,7,10,12], 3, "botsurf")  
gmsh.model.addPhysicalGroup(1, [1,3,4,2,15,20,24], 4, "fixedup")  
gmsh.model.addPhysicalGroup(2, [5], 1,"midsuf")  
gmsh.model.addPhysicalGroup(2, [34], 2, "topsuf")  
gmsh.model.addPhysicalGroup(2, [6], 3, "botsurf")  
gmsh.model.addPhysicalGroup(2, [1, 25], 4, "fixedup")  
gmsh.model.addPhysicalGroup(3, [1,2], 1, "Volume")  


output_file = joinpath(dirname(@__FILE__), "model_test.msh")
gmsh.write(output_file)

 # Launch the GUI to see the results:
if !("-nopopup" in ARGS)
      # gmsh.fltk.run()
end

Gmsh.finalize()