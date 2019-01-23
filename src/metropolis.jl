"""
A single sweep of each link on lattice with multi-hit Metropolis update (hit=10)
"""
function metropolis_update!(epsilon::Float64, lattice::Lattice)
    multihit = 1 # number of hits per link update
    maxphi = 100 # arbitary upperbound
    @assert lattice.quenched == true "Lattice has to be quenched to use "*
                                     "metropolist update!"
    @assert epsilon < maxphi "don't choose epsilon to be too large! (<0.5)"

    # Action contribution by link specified by i, mu
    function SGimu(i, lattice, mu)
        @assert (mu == 1 || mu == 2) "mu has to be 1 (x-direction) or 2 (t-direction)"
        com = cos(lattice.anglex[i] + lattice.anglet[lattice.rightx[i]] -
                  lattice.anglex[lattice.upt[i]] - lattice.anglet[i])
        if mu == 1 # x-direction
            ret = com + cos(-lattice.anglet[lattice.downt[i]] + 
                            lattice.anglex[lattice.downt[i]] +
                            lattice.anglet[lattice.rightx[lattice.downt[i]]] -
                            lattice.anglex[i])
        end
        if mu == 2 # t-direction
            ret = com + cos(lattice.anglex[lattice.leftx[i]] + 
                            lattice.anglet[i] -
                            lattice.anglex[lattice.leftx[lattice.upt[i]]] -
                            lattice.anglet[lattice.leftx[i]])
        end
        return -lattice.beta * ret
    end

    accpt = 0 # total accepted updates
    upiter = 0 # total update iterations
    for i in 1:lattice.ntot
        for ihit in 1:multihit
            accpt += 1
            upiter += 1
            # First update links in x direction
            oldS = SGimu(i, lattice, 1) # this use angles ONLY so no need to sync links yet
            theta = 2 * (rand01() - 0.5) * epsilon
            lattice.anglex[i] = lattice.anglex[i] + theta
            dMeS = SGimu(i, lattice, 1) - oldS
            # Reject
            if rand01() > exp(-dMeS)  
                lattice.anglex[i] = lattice.anglex[i] - theta
                accpt -= 1
            end
        end
        sync!(lattice)
        # Do not update t-link at tmax for sublattice updates
        # Important for domain decomposition
        if (lattice.corr_indx[i][2] == lattice.nt && lattice.boundary_cond == "time dirichlet")
            sync!(lattice)
            continue
        else
            for ihit in 1:multihit
                # Then update links in t direction (besides freezing domain on the boundary)
                accpt += 1
                upiter += 1
                oldS = SGimu(i, lattice, 2)
                theta = 2 * (rand01() - 0.5) * epsilon
                lattice.anglet[i] = lattice.anglet[i] + theta
                dMeS = SGimu(i, lattice, 2) - oldS
                # Reject
                if rand01() > exp(-dMeS)
                    lattice.anglet[i] = lattice.anglet[i] - theta
                    accpt -= 1
                end
            end # multihit
        end
        sync!(lattice)
    end # i site
    sync!(lattice)
    return accpt/upiter # return acceptance rate for a single sweep
end