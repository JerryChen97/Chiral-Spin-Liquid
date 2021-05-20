import tenpy
from tenpy.networks.terms import TermList

import numpy as np

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

Lx,Ly=2,4
L=2*Lx*Ly

Sp = 'Sp'
Sm = 'Sm'
Sz = 'Sz'

# 16 possible local couplings. NN and nNN hoping, SxSx+SySy vs. SzSz, two sublattices
op_list=[[Sp, Sm, 1, 0], [Sm, Sp, 1, 0], [Sp, Sm, 1, 1], [Sm, Sp, 1, 1],\
         [Sz, Sz, 1, 0], [Sz, Sz, 1, 0], [Sz, Sz, 1, 1], [Sz, Sz, 1, 1],\
         [Sp, Sm, 2, 0], [Sm, Sp, 2, 0], [Sp, Sm, 2, 1], [Sm, Sp, 2, 1],\
         [Sz, Sz, 2, 0], [Sz, Sz, 2, 0], [Sz, Sz, 2, 1], [Sz, Sz, 2, 1],\
         [Sp, Sm, 2*Ly-1, 0], [Sm, Sp, 2*Ly-1, 0], [Sp, Sm, 2*Ly-1, 1], [Sm, Sp, 2*Ly-1, 1],\
         [Sz, Sz, 2*Ly-1, 0], [Sz, Sz, 2*Ly-1, 0], [Sz, Sz, 2*Ly-1, 1], [Sz, Sz, 2*Ly-1, 1],\
         [Sp, Sm, 3, 0], [Sm, Sp, 3, 0], [Sp, Sm, 3, 1], [Sm, Sp, 3, 1],\
         [Sz, Sz, 3, 0], [Sz, Sz, 3, 0], [Sz, Sz, 3, 1], [Sz, Sz, 3, 1]]

h_list = []

for i in range(L):
    tl = []
    if op_list[2*i][3]==0:
        for m in range(L):
            if m%2==0:
                if m%(2*Ly) +op_list[2*i][2] < 2*Ly:
#                     exp+=psi.expectation_value_term([(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2])])\
#                         +psi.expectation_value_term([(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]  )])
                    tl += [
                        [(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2])], 
                        [(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]  )]
                    ]
                else:
#                     exp+=psi.expectation_value_term([(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2]-2*Ly)])\
#                         +psi.expectation_value_term([(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]-2*Ly  )])
                    tl += [
                        [(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2]-2*Ly)],
                        [(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]-2*Ly  )]
                    ]
                    
    else:
        if op_list[2*i][2]%2==0:
            for m in range(L):
                if m%2 !=0:
                    if m%(2*Ly) +op_list[2*i][2] < 2*Ly:

#                         exp+=psi.expectation_value_term([(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2])])\
#                             +psi.expectation_value_term([(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]  )])
                        tl += [
                            [(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2])],
                            [(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]  )]
                        ]
                    else:
#                         exp+=psi.expectation_value_term([(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2]-2*Ly)])\
#                             +psi.expectation_value_term([(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]-2*Ly  )])
                        tl += [
                            [(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2]-2*Ly)],
                            [(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]-2*Ly  )],
                        ]
                    
        else: 
            for m in range(L):
                if m%2 !=0:

                    if m%(2*Ly) +op_list[2*i][2] + 2*Ly-2 < 4*Ly:
                        tl += [
                            [(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2]+2*Ly-2)],
                            [(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]+2*Ly-2  )],
                        ]
#                         exp+=psi.expectation_value_term([(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2]+2*Ly-2)])\
#                             +psi.expectation_value_term([(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]+2*Ly-2  )])
                    else:
#                         exp+=psi.expectation_value_term([(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2]-2  )])\
#                             +psi.expectation_value_term([(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]-2  )])
                        tl += [
                            [(op_list[2*i][0] , m),(op_list[2*i][1], m+op_list[2*i][2]-2  )],
                            [(op_list[2*i+1][0] , m),(op_list[2*i+1][1], m+op_list[2*i+1][2]-2  )],
                        ]
    h_list.append(TermList(terms=tl))