include("./lattice.jl")
include("./spinor.jl")
include("./dirac.jl")
include("./hmc_types.jl")
include("./hmc_wilson_forces.jl")


function leapfrog!(p::HMCMom, pf::PseudoFermion, nsteps::Int64, 
                  dtau::Float64, quenched::Bool, lattice::Lattice)
    updatemom!(p, pf, dtau./2, quenched, lattice)
    for istep in 1:nsteps
        println("Leapfrog steps: $istep/$nsteps")
        updategauge!(p, dtau, lattice)
        updatemom!(p, pf, dtau, quenched, lattice)
    end
    updatemom!(p, pf, dtau./2, quenched, lattice)
end

function updatemom!(p::HMCMom, pf::PseudoFermion, dtau::Float64, quenched::Bool,
                    lattice::Lattice)
    if quenched == false
        # pforce_common involves inversion so we want to do it outside of the loop
        # lhs = left-hand-side of dot product and psi =  D^{-1}phi for phi = pf field
        lhs, psi = pforce_common(pf, lattice)
    end
    for i in 1:lattice.ntot
        if quenched == false
            p.gpx[i] = p.gpx[i] - dtau*(dSG1(i, lattice) + pforce1(i, pf, lattice, psi, lhs))
            p.gpt[i] = p.gpt[i] - dtau*(dSG2(i, lattice) + pforce2(i, pf, lattice, psi, lhs))
        else
            p.gpx[i] = p.gpx[i] - dtau*(dSG1(i, lattice))
            p.gpt[i] = p.gpt[i] - dtau*(dSG2(i, lattice))
        end
    end
end

function updategauge!(p::HMCMom, dtau::Float64, lattice::Lattice)
    for i in 1:lattice.ntot
        lattice.anglex[i] = lattice.anglex[i] + dtau*p.gpx[i]
        lattice.anglet[i] = lattice.anglet[i] + dtau*p.gpt[i]
        # Don't forget to update the link as well
        lattice.linkx[i] = cos(lattice.anglex[i]) + im*sin(lattice.anglex[i])
        lattice.linkt[i] = cos(lattice.anglet[i]) + im*sin(lattice.anglet[i])
    end
end

function test_leapfrog()
    lattice = Lattice(3, 3, 0.1, 0.1)
    pf = PseudoFermion(lattice, gamma5_Dslash_wilson)
    p = HMCMom(lattice)
    leapfrog!(p, pf, 10, 1.0, lattice)
end
#test_leapfrog()