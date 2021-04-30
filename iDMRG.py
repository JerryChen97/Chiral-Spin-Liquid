from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
import numpy as np
from tenpy.models.lattice import Lattice
import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.networks.mps import MPS
from tenpy.algorithms.truncation import svd_theta
from tenpy.linalg.np_conserved import Array
from tenpy.models.model import CouplingModel
from tenpy.models.model import CouplingMPOModel 


# Saving Files
import h5py
from tenpy.tools import hdf5_io


dmrg_params = {
    'mixer': True,
    'trunc_params': {'chi_max': 1000,
                     'svd_min': 1.e-8},
    'max_E_err': 1.e-8,
    'max_S_err': 1.e-7,
    'N_sweeps_check': 2,
    'max_sweeps': 100,
    'verbose': 1,
    'chi_list': {0: 100, 10: 1000},
}

model_params = {
    'Lx': 1, 'Ly': 8, 'bc':'periodic',# Ly is set below
    'bc_MPS': 'infinite',
    'verbose': 1,
    'cons_N': 'N',
    'cons_Sz': 'Sz',
    't': 1,
    'td': 0.5j,
}



class twosites(Lattice):
    def __init__(self, Lx, Ly, site, **kwargs):
     
       
        NN = [(0, 1, np.array([0, 0])), (1, 0, np.array([1, 0])), (0, 0, np.array([0, 1]))]
        NNa = [(1, 1, np.array([0, 1]))]
        nNN = [(0, 1, np.array([0, 1])), (1, 0, np.array([0, 1])),(1, 0, np.array([1, -1])), (0, 1, np.array([-1, -1]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', NNa)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nNN)
        Lattice.__init__(self, [Lx, Ly], site, **kwargs)




class FermionicChiral(CouplingMPOModel):

    #default_lattice = doublesites
    #force_default_lattice = True

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site


    def init_lattice(self, model_params):
        Lx = model_params.get('Lx',1)
        Ly = model_params.get('Ly',8)
        fs = self.init_sites(model_params)
        bc = model_params.get('bc','periodic')
        bc_MPS = model_params.get('bc_MPS','infinite')
       

        lat = twosites(Lx, Ly, [fs,fs],bc=bc, bc_MPS=bc_MPS)
        return lat

    def init_terms(self, model_params):
        
        t = model_params.get('t', 1.)
        td = model_params.get('td', 0.5j)
       

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(t , u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(t , u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
            
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
            
        for u1, u2, dx in self.lat.pairs['next_next_nearest_neighbors']:
            self.add_coupling(td, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(td, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)

      
            
            
            
CSLnoGPO=FermionicChiral(model_params)

psi = MPS.from_lat_product_state(CSLnoGPO.lat, [[["up","down"]]],bc=CSLnoGPO.lat.bc_MPS)

            

            
            
eng = TwoSiteDMRGEngine(psi, CSLnoGPO, dmrg_params)



E, psi = eng.run()

print(E)
Lx = model_params['Lx']
Ly = model_params['Ly']
chimax = dmrg_params['trunc_params']['chi_max']
file_name = f'iDMRPG_CSL_Lx_{Lx}_Ly_{Ly}_chimax_{chimax}.h5'
with h5py.File(file_name, 'w') as f:
    hdf5_io.save_to_hdf5(f, psi)
    
