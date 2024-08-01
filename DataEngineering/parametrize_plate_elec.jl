using Gmsh: Gmsh, gmsh
Gmsh.initialize()

gmsh.model.add("parametrize_plate_elec")

# parameters
const Lx = 40;      # beam length
const Ly = 8;       # beam width
const Lz = 0.4;     # beam thickness


const divmeshx = 10; # X element size in each electrode
const divmeshy = 4; # Y element size in each  electrode
const divmeshz = 2; # Z element size in each  electrode

const elecdivx = 5; # number of electrodes in X
const elecdivy = 2; # number of electrodes in Y

const lc = 1.0; # characteristic length for meshing

Lex = Lx / elecdivx
Ley = Ly / elecdivy


geo = gmsh.model.geo


function generatepoints!(geo, elecdivx, elecdivy, Lex, Ley, initialtag=1)
    ptag = initialtag
    for i in 1:elecdivx+1
        for j in 1:elecdivy+1
            geo.addPoint((i - 1) * Lex, (j - 1) * Ley, 0, lc, ptag)
            ptag += 1
        end
    end
end

function generatelines!(geo, elecdivx, elecdivy, initialtag=1)
    ptag = initialtag
    for i in 0:elecdivx
        for j in 0:elecdivy
            if j < elecdivy
                # @show ptag+i, ptag+i+1, ptag
                geo.addLine(ptag + i, ptag + i + 1, ptag)
                if i < elecdivx
                    # @show ptag+i, ptag+i+(elecdivy+1), i+ptag+(elecdivy*(elecdivx+1))
                    geo.addLine(ptag + i, ptag + i + (elecdivy + 1), i + ptag + (elecdivy * (elecdivx + 1)))
                end
                ptag += 1
            else
                # @show ptag+i, ptag+i+(elecdivy+1), i+ptag+((elecdivy)*(elecdivx+1))
                if i < elecdivx
                    geo.addLine(ptag + i, ptag + i + (elecdivy + 1), i + ptag + ((elecdivy) * (elecdivx + 1)))
                end
            end
        end
    end


end

function generatecurveloops!(geo, elecdivx, elecdivy, initialtag=1)
    nprev = elecdivy * (elecdivx + 1)
    ptag = initialtag
    for i in 0:(elecdivx-1)
        for j in 0:(elecdivy-1)
            # @show nprev+ptag+i,ptag+elecdivy,-(nprev+ptag+1+i),-ptag
            geo.addCurveLoop([nprev + ptag + i, ptag + elecdivy, -(nprev + ptag + 1 + i), -ptag], ptag)
            ptag += 1

        end
    end

end
generatepoints!(geo, elecdivx, elecdivy, Lex, Ley)
generatelines!(geo, elecdivx, elecdivy)
generatecurveloops!(geo, elecdivx, elecdivy)
geo

nprev = elecdivy * (elecdivx + 1)

for i in 1:nprev
    geo.mesh.setTransfiniteCurve(i, divmeshy)
end
for i in nprev+1:nprev+(elecdivx*(elecdivy+1))
    geo.mesh.setTransfiniteCurve(i, divmeshx)
end

bodies_mid = []
bodies_up = []

for i in 1:elecdivx*elecdivy
    geo.addPlaneSurface([i], i)
    geo.mesh.setTransfiniteSurface(i)
    geo.mesh.setRecombine(2, i)
    body = gmsh.model.geo.extrude([(2, i)], 0.0, 0.0, Lz, [divmeshz - 1], [1], true)
    body_= gmsh.model.geo.extrude([(2, body[1][2])], 0.0, 0.0, Lz, [divmeshz - 1], [1], true)
    push!(bodies_mid,body)
    push!(bodies_up, body_)
end
@show bodies_mid[2][1][2]

fixedsurf=[]
for i in 1:elecdivy
    push!(fixedsurf, bodies_mid[i][end][2])
    push!(fixedsurf, bodies_up[i][end][2])
end
 
# geo.mesh.setTransfiniteVolume(2)
gmsh.model.geo.synchronize()



function getattached(surfnum) 

    linesattached=[]
    for i in 1:size(surfnum)[1]
        up, linesattached_ = gmsh.model.getAdjacencies(2, surfnum[i])
        append!(linesattached, linesattached_)
    end 
    linesattached=unique(linesattached)
    
    pointsattached=[]
    for i in 1:size(linesattached)[1]
        up, pointsattached_ = gmsh.model.getAdjacencies(1, linesattached[i])
        append!(pointsattached, pointsattached_)
    end
    pointsattached=unique(pointsattached)

    return pointsattached, linesattached
end

# # Generate mesh
    gmsh.model.mesh.generate(3) 

   # Inicializar un array vacío para almacenar los segundos elementos
topsurf = []
midsurf = []
botsurf = []
botvol  = []
topvol  = []
# Bucle for que extrae los segundos elementos
for i in  1:elecdivx*elecdivy
    topsurf_ = bodies_up[i][1][2]
    midsurf_ = bodies_mid[i][1][2]
    botvol_  = bodies_mid[i][2][2]
    topvol_  = bodies_up[i][2][2]
    push!(midsurf, midsurf_)
    push!(topsurf, topsurf_)
    push!(botsurf, i)
    push!(botvol, botvol_)
    push!(topvol, topvol_)

end

for i in 1:elecdivx*elecdivy
    ibot=i
    imid=elecdivx*elecdivy+i
    isup=2*(elecdivx*elecdivy)+i

    pfixed, lfixed= getattached([topsurf[i]]) 
    gmsh.model.addPhysicalGroup(2, [topsurf[i]], isup, "topsuf_$isup")  
    gmsh.model.addPhysicalGroup(1, lfixed, isup, "topsuf_$isup")  
    gmsh.model.addPhysicalGroup(0, pfixed, isup, "topsuf_$isup")  

    pfixed, lfixed= getattached([midsurf[i]]) 
    gmsh.model.addPhysicalGroup(2, [midsurf[i]], imid, "midsuf_$imid")  
    gmsh.model.addPhysicalGroup(1, lfixed, imid, "midsuf_$imid")  
    gmsh.model.addPhysicalGroup(0, pfixed, imid, "midsuf_$imid")  

    pfixed, lfixed= getattached([botsurf[i]]) 
    gmsh.model.addPhysicalGroup(2, [botsurf[i]], ibot, "botsuf_$ibot")  
    gmsh.model.addPhysicalGroup(1, lfixed, ibot, "botsuf_$ibot")  
    gmsh.model.addPhysicalGroup(0, pfixed, ibot, "botsuf_$ibot")  

end
 
pfixed, lfixed= getattached(fixedsurf) 
gmsh.model.addPhysicalGroup(2, fixedsurf, 3*elecdivx*elecdivy+1, "fixed") 
gmsh.model.addPhysicalGroup(1, lfixed, 3*elecdivx*elecdivy+1, "fixed") 
gmsh.model.addPhysicalGroup(0, pfixed, 3*elecdivx*elecdivy+1, "fixed") 

gmsh.model.addPhysicalGroup(3, botvol, 1, "Volbot")  
gmsh.model.addPhysicalGroup(3, topvol, 2, "Volup")  

 
 

output_file = joinpath(dirname(@__FILE__), "parametrize_plate_elec_complex.msh")
gmsh.write(output_file)

# Launch the GUI to see the results:
#if !("-nopopup" in ARGS)
#    gmsh.fltk.run()
#end

Gmsh.finalize()