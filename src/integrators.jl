include("./lattice.jl")
include("./spinor.jl")
include("./dirac.jl")
include("./solvers.jl")
include("./hmc.jl")


function leapfrog(lattice::Lattice, p::Mom, nsteps::Int64, dtau::Float64)

end

function updatemom!(lattice::Lattice, p::Mom, dtau::Float64)

end

function updategauge!(lattice:Lattice, p::Mom, dtau::Float64)
    for i in 1:lattice.ntot
        lattice.anglex[i] = lattice.anglex[i] + dtau*p.gpx[i]
        lattice.anglet[i] = lattice.anglet[i] + dtau*p.gpt[i]
        # Don't forget to update the link as well
        lattice.linkx[i] = cos(lattice.anglex[i]) + im*sin(lattice.anglex[i])
        lattice.linkt[i] = cos(lattice.anglet[i]) + im*sin(lattice.anglet[i])
    end
end