import numpy as np

from itertools import permutations

import tenpy
from tenpy.models.lattice import Lattice
import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.networks.terms import TermList

from tenpy.tools.hdf5_io import save, load

# Log
import logging.config
conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'custom': {'format': '%(levelname)-8s: %(message)s'},
    },
    'handlers': {
        'to_file': {
            'class': 'logging.FileHandler',
            'filename': f'Chiral.log',
            'formatter': 'custom',
            'level': 'INFO',
            'mode': 'a',
            },
        'to_stdout': {
            'class': 'logging.StreamHandler',
            'formatter': 'custom',
            'level': 'INFO',
            'stream': 'ext://sys.stdout',
            },
    },
    'root': {
        'handlers': ['to_stdout', 'to_file'], 
        'level': 'DEBUG',
    },
}
logging.config.dictConfig(conf)

# Print the version of tenpy
print("Current TeNPy version: ", tenpy.version.version)

def mul(self, other):
    terms = []
    strength = []
    for termlist1 in self:
        for termlist2 in other:
            termlist = termlist1[0] + termlist2[0]
            terms.append(termlist)
            strength.append(termlist1[1] * termlist2[1])
    return TermList(terms, strength)
TermList.__mul__ = mul

class QCM:
    def __init__(self, psi, h_list):
        self.psi = psi.copy() # MPS
        self.h_list = h_list # the basis vector h_a, a list of TermList
        
        self.calc_average_ha() # the average, i.e. expectation value of each basis
        self.calc_average_hahb() # the cross terms
        self.calc_QCM()
        
    def calc_average_ha(self):
        """
            Calculate the expectation values of each basis Hamiltonian,
            by seeing each as a sum of local operators and then directly calculating
            the expectation value
        """
        self.average_ha = [self.psi.expectation_value_terms_sum(h)[0] for h in self.h_list]
    def calc_average_hahb(self):
        """
            The cross values...
        """
        n = len(self.h_list)
        self.ab_matrix = np.zeros((n, n), dtype='complex64')
        for i in range(n):
            for j in range(n):
                self.ab_matrix[i, j] = self.psi.expectation_value_terms_sum(self.h_list[i] * self.h_list[j])[0]
    def calc_QCM(self):
        n = len(self.h_list)
        mat = self.ab_matrix.copy()
        for i in range(n):
            for j in range(n):
                mat[i, j] -= self.average_ha[i]*self.average_ha[j]
        self.qcm = mat


psi = load('projected_psi_1000.h5')

def symmetrize(op_list, mode='full'):
    # Here op is a temporary var
    op_name_list = [op[0] for op in op_list]
    op_pos_list = [op[1] for op in op_list]
    l = len(op_list)
    if mode == 'cyclic':
        new_op_list = []
        for off_set in range(l):
            op = [(op_name_list[(i+off_set)%l], op_pos_list[i]) for i in range(l)]
            new_op_list.append(op)
        return new_op_list
    elif mode == 'full':
        
        new_op_list = []
        
        all_perm = (list(permutations(op_name_list)))
        for perm_name_list in all_perm:
            op = [(perm_name_list[i], op_pos_list[i]) for i in range(l)]
            new_op_list.append(op)
        return new_op_list
        
    raise NotImplemented()
# op_list_test = [('Sp', 0), ('Sm', 1), ('Sz', 2)]
# symmetrize(op_list_test)

class twosites(Lattice):
    def __init__(self, Lx, Ly, site, **kwargs):
        
       
        basis = np.array(([1., 0.], [0., 1.]))
        delta = np.array([1., 0.])
        pos = (-delta / 4., delta / 4.)
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
     
       
        """
            The second is for the pairs' definition
        """
        NN = [(0, 1, np.array([0, 0])), (1, 0, np.array([1, 0])), (0, 0, np.array([0, 1]))]
        NNa = [(1, 1, np.array([0, 1]))]
        nNN = [(0, 1, np.array([0, 1])), (1, 0, np.array([0, 1])),(1, 0, np.array([1, -1])), (0, 1, np.array([-1, -1]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', NNa)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nNN)
       
        
      
     
        Lattice.__init__(self, [Lx, Ly], site, **kwargs)
        
f = SpinHalfFermionSite()
lat = twosites(Lx=1, Ly=8, site=[f, f], bc='periodic', bc_MPS='infinite')

# 8 different spin chirality operators (different sublattice/orientation)
Chirality = [
    [(np.array([0, 0]), 0), (np.array([0, 0]), 1), (np.array([0,-1]), 0)],
    [(np.array([0, 0]), 0), (np.array([0, 0]), 1), (np.array([0,-1]), 1)],
    [(np.array([0, 0]), 0), (np.array([0, 0]), 1), (np.array([0, 1]), 0)],
    [(np.array([0, 0]), 0), (np.array([0, 0]), 1), (np.array([0, 1]), 1)],
    [(np.array([0, 0]), 1), (np.array([1, 0]), 0), (np.array([0,-1]), 1)],
    [(np.array([0, 0]), 1), (np.array([1, 0]), 0), (np.array([1,-1]), 0)],
    [(np.array([0, 0]), 1), (np.array([1, 0]), 0), (np.array([0, 1]), 1)],
    [(np.array([0, 0]), 1), (np.array([1, 0]), 0), (np.array([1, 1]), 0)],
]


op_list1=['Sp', 'Sm', 'Sz']
op_list2=['Sm', 'Sz', 'Sp']
op_list3=['Sz', 'Sp', 'Sm']

op_lists = [op_list1, op_list2, op_list3]


#! Create the basis vector
h_base = []

for dx_u in Chirality:
    
    op_list = ['Sp', 'Sm', 'Sz']
    ops = list(zip(op_list, dx_u))
#         print(dx_u)
#         print(ops)
    ops = [(op[0], op[1][0], op[1][1]) for op in ops]
    mps_ijkl, _, _ = (lat.possible_multi_couplings(ops))
    h = [] 
    for ids in mps_ijkl:
        h += symmetrize([(op_list[i], ids[i]) for i in range(len(ids))])
#     print(h)
# #     print(TermList(terms=h))
#     print('')
    h_base.append(TermList(terms=h))
        
# 8 different two site couplings
Twositecoupling = [
    [(np.array([0, 0]), 0), (np.array([0, 0]), 1)],
    [(np.array([0, 0]), 1), (np.array([1, 0]), 0)],
    [(np.array([0, 0]), 0), (np.array([0, 1]), 0)],
    [(np.array([0, 0]), 1), (np.array([0, 1]), 1)],
    [(np.array([0, 0]), 0), (np.array([0, 1]), 1)],
    [(np.array([0, 0]), 1), (np.array([0, 1]), 0)],
    [(np.array([0, 0]), 1), (np.array([1, 1]), 0)],
    [(np.array([0, 0]), 1), (np.array([1,-1]), 0)],
]

for dx_u in Twositecoupling:
    
    op_list = ['Sp', 'Sm']
    ops = list(zip(op_list, dx_u))
#         print(dx_u)
#         print(ops)
    ops = [(op[0], op[1][0], op[1][1]) for op in ops]
    mps_ijkl, _, _ = (lat.possible_multi_couplings(ops))
    h = [] 
    for ids in mps_ijkl:
        h += symmetrize([(op_list[i], ids[i]) for i in range(len(ids))])
#     print(h)
# #     print(TermList(terms=h))
#     print('')
    h_base.append(TermList(terms=h))
    
    
    op_list = ['Sz', 'Sz']
    ops = list(zip(op_list, dx_u))
#         print(dx_u)
#         print(ops)
    ops = [(op[0], op[1][0], op[1][1]) for op in ops]
    mps_ijkl, _, _ = (lat.possible_multi_couplings(ops))
#     h = [] 
    for ids in mps_ijkl:
        h += symmetrize([(op_list[i], ids[i]) for i in range(len(ids))])
#     print(h)
# #     print(TermList(terms=h))
#     print('')
    h_base.append(TermList(terms=h))            

qcm = QCM(psi, h_base)
save(qcm, 'qcm.h5')
