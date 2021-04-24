import pythtb as tb
import numpy as np
from numpy import linalg as LA


#set and solve the parton band model
t, td = 1., 0.5j
Lx ,Ly =6 , 4
L = Lx * Ly * 2
chimax = 800
cutoff = 1e-10


# define lattice vectors
lat=[[2.0,0.0],[0.0,1.0]]
# define coordinates of orbitals
orb=[[0.0, 0.0], [0.5, 0.0]]
# make two dimensional tight-binding model
parton=tb.tb_model(2,2,lat,orb)
# set model parameters
# add first neighbour hoppings
parton.set_hop(t, 0, 1, [0, 0])
parton.set_hop(t, 1, 0, [1, 0])
parton.set_hop(t, 0, 0, [0, 1])
parton.set_hop(-t, 1, 1, [0, 1])
# add second neighbour hoppings
parton.set_hop(td, 0, 1, [0, 1])
parton.set_hop(td, 1, 0, [0, 1])
parton.set_hop(-td, 1, 0, [1, 1])
parton.set_hop(-td, 0, 1, [-1,1])


#We first work with a finite system
# cutout finite model first along direction x without PBC
tmp_parton=parton.cut_piece(Lx,0,glue_edgs=False)
# cutout also along y direction with PBC
fin_parton=tmp_parton.cut_piece(Ly,1,glue_edgs=True)
# solve finite model
(evals,evecs) =fin_parton.solve_all(eig_vectors=True)
# computes hybrid Wannier centers (and functions) for the filled band. This procedure reduces truncation errors.
# We work with Wannier basis to reduce long range entanglement and truncation error 
filled=evecs[0:Lx*Ly,:]
pos_matx=fin_parton.position_matrix(evecs[:Lx*Ly],0)
pos_maty=fin_parton.position_matrix(evecs[:Lx*Ly],1)
pos_mat=Ly*pos_maty+pos_matx
Wanniercenter, Wannier= LA.eig(pos_mat)
parton_hwf=np.matmul(Wannier.transpose() ,filled)


# Two colors of partons, labeled by spin index. We add spin up parton first.
import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.networks.mps import MPS
from tenpy.algorithms.truncation import svd_theta



#1. Represent fermion vacuum as MPS: |0>=|0000...>
site = SpinHalfFermionSite()
psi = MPS.from_product_state([site]*L, ['empty']*L, 'finite')



Aml = parton_hwf

chinfo, p_leg = site.leg.chinfo, site.leg


def Ws(Aml, m):
    "MPO for d_m^\dag (spin up)"
    Cdu, Id = [site.get_op(op) for op in ('Cdu', 'Id')]
    Wm = []
    wL_leg = npc.LegCharge.from_qflat(chinfo,  [op.qtotal for op in [Cdu, Id]] , qconj=+1) # set charge on the leg
    for l in range(L):
        Wml_grid = [[Id, None],
                    [Aml[m,l]*Cdu, Id]]
        Wml = npc.grid_outer(Wml_grid, [wL_leg, wL_leg.conj()],grid_labels=['wL', 'wR']) # wL, wR, p, p*
        Wm.append(Wml)
    # Boundary case
    Wm[0] = Wm[0][1:2,:] # second row
    Wm[-1] = Wm[-1][:, 0:1] # first column
    return Wm

#4. Apply MPO to MPS and then perform SVD compressions: d_m^dag |0>
for m in range(L//2): # v=1
    Wm = Ws(Aml, m)
    print(m)
    for i in range(L):
        B = psi.get_B(i, form='B')
        Wml = Wm[i]
        B = npc.tensordot(Wml, B, ['p*', 'p']) # wL wR p vL vR 
        B.itranspose(['vL', 'wL', 'p', 'vR', 'wR'])
        B = B.combine_legs([('vL', 'wL'), ('vR', 'wR')], qconj=[1, -1])
        B.iset_leg_labels(['vL', 'p', 'vR'])
        psi.set_B(i, B, form=None)
    psi.canonical_form() # to BBB...

    # SVD compression
    for i in range(L-1):
        theta = psi.get_theta(i, n=2)
        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[1, -1]) # vL.p0 p1.vR
        A0, S1, B1, error, renorm = svd_theta(theta, {'svd_min': cutoff,
                                                      'chi_max': chimax})
        psi.set_B(i, A0.split_legs().replace_label('p0', 'p'), form='A')
        psi.set_SR(i, S1)
        psi.set_B(i+1, B1.split_legs().replace_label('p1', 'p'), form='B')
        
        
        




#Then put on spin down partons
site = SpinHalfFermionSite()



Aml = parton_hwf

chinfo, p_leg = site.leg.chinfo, site.leg


def Ws(Aml, m):
    "MPO for d_m^\dag (spin down)"
    Cdd, Id = [site.get_op(op) for op in ('Cdd', 'Id')]
    Wm = []
    wL_leg = npc.LegCharge.from_qflat(chinfo, [op.qtotal for op in [Cdd, Id]], qconj=+1) # charge of spin down objects
    for l in range(L):
        Wml_grid = [[Id, None],
                    [Aml[m,l]*Cdd, Id]]
        Wml = npc.grid_outer(Wml_grid, [wL_leg, wL_leg.conj()],grid_labels=['wL', 'wR']) # wL, wR, p, p*
        Wm.append(Wml)
    # Boundary case
    Wm[0] = Wm[0][1:2,:] # second row
    Wm[-1] = Wm[-1][:, 0:1] # first column
    return Wm

#4. Apply MPO to MPS and then perform SVD compressions: d_m^dag |0>
for m in range(L//2): # v=1
    Wm = Ws(Aml, m)
    print(m)
    for i in range(L):
        B = psi.get_B(i, form='B')
        Wml = Wm[i]
        B = npc.tensordot(Wml, B, ['p*', 'p']) # wL wR p vL vR
        B.itranspose(['vL', 'wL', 'p', 'vR', 'wR'])
        B = B.combine_legs([('vL', 'wL'), ('vR', 'wR')], qconj=[1, -1])
        B.iset_leg_labels(['vL', 'p', 'vR'])
        psi.set_B(i, B, form=None)
    psi.canonical_form() # to BBB...

    # SVD compression
    for i in range(L-1):
        theta = psi.get_theta(i, n=2)
        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[1, -1]) # vL.p0 p1.vR
        A0, S1, B1, error, renorm = svd_theta(theta, {'svd_min': cutoff,
                                                      'chi_max': chimax})
        psi.set_B(i, A0.split_legs().replace_label('p0', 'p'), form='A')
        psi.set_SR(i, S1)
        psi.set_B(i+1, B1.split_legs().replace_label('p1', 'p'), form='B')
        
        
        
        

# Gutzwiller projection to physical Hilbert space
from tenpy.linalg.np_conserved import Array
import numpy as np

chinfo, p_leg = site.leg.chinfo, site.leg

def GPO():
    "MPO for Gutzwiller projection"
    GPO = []
    for l in range(L):
        GPOls = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]])
        GPOl=Array.from_ndarray(GPOls, [p_leg, p_leg.conj()]) # p, p*
        GPOl.iset_leg_labels(['p', 'p*'])
        GPO.append(GPOl)
    return GPO

gpos = GPO()
#4. Apply the GPO
for l in range(L): # GPO
    GPOl = gpos[l]
    print(l)
    B = psi.get_B(l, form='B')
    B = npc.tensordot(GPOl, B, ['p*', 'p']) # wL wR p vL vR
    psi.set_B(l, B, form=None)
    psi.canonical_form() # to BBB...
    
