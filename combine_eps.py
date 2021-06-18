#!/usr/bin/env python

# written by Jack McArthur, last edited 1/24/2021
# supervised by Prof. Diana Qiu

# this program takes inverse epsilon matrices for two sublattices, .h5 format, 
# and writes an inverse epsilon matrix representing the superposition of the two lattices.
# the two sublattice eps^-1 .h5's must have the same unit cell parameters and G vector grid.

import numpy as np
import h5py

def matx_combine(matx1, matx2, qpt, N_g, vcoul, nfreq=1):
    # finds Kohn-Sham polarizabilities (chi^0) of matx1 and matx2, 
    # finds chi^0 of their superposition, and returns eps_out from that chi^0

    # matx1/2: the N_qpt x N_freq x N_G x N_G matrix from epsmat.h5 of eps^-1_GG' for all qpoints sampled
    # qpt is the index of the qpt in question
    # N_g comes from f['eps_header']['gspace']['nmtx']
    # vcoul comes from f['eps_header']['gspace']['vcoul']
    # nfreq is the number of frequencies to do SSA on for each qpoint; defaults to 1
    # eventual output, eps_inv is a list of length n_freq of N_G x N_G matrices for the qpoint in question
    # eps_inv[n] is the N_G x N_G matrix for the nth frequency at qpoint qpt
    eps_inv = []

    # helpful later
    v_G = np.diag(vcoul[qpt,:N_g])
    v_G_inv = np.diag(1. / vcoul[qpt,:N_g])

    # loop over frequencies
    for freq in range(nfreq):
        chi0_GG1 = v_G_inv @ (-np.linalg.inv(matx1[qpt,freq][:N_g,:N_g].conj().T) + np.identity(N_g))
        chi0_GG2 = v_G_inv @ (-np.linalg.inv(matx2[qpt,freq][:N_g,:N_g].conj().T) + np.identity(N_g))

        chi0_out = chi0_GG1 + chi0_GG2
        eps_inv.append(np.linalg.inv(np.identity(N_g) - v_G @ chi0_out).conj().T)

    return eps_inv

if __name__=='__main__':

    import sys

    if (len(sys.argv) != 4) and (len(sys.argv) != 5):
        print("Usage: combine_eps.py filename1.h5 filename2.h5 outputname.h5 exactout.h5=None")
        print("\'exactout.h5\' is optional and will be used only to take header information")
        sys.exit()

    log_file = open(f'{sys.argv[3][:-3]}.log',"w")
    old_stdout = sys.stdout
    sys.stdout = log_file

    # Open old and new epsmat files and copy header info
    print(f"opening {sys.argv[1]}...\n")
    f_1 = sys.argv[1]
    f1 = h5py.File(f_1 , 'r')
    print(f"opening {sys.argv[2]}...\n")
    f_2 = sys.argv[2]
    f2 = h5py.File(f_2 , 'r')

    nq = f1['eps_header/qpoints/nq'][()]
    nfreq = f1['eps_header/freqs/nfreq'][()]
    print(f"there are {nq} q-points with matrices for {nfreq} frequencies each")

    # qpts: list of qpts stored in f['mats']['matrix']
    qpts = f1['eps_header']['qpoints']['qpts']
    # mat_size: mat_size[qpt_id] gives size of matx[qpt_id], as all eps^-1_GG'(q)
    # have the same shape (n x n), but some have empty rows at the end
    mat_sizes = f1['eps_header']['gspace']['nmtx']
    # vcoul: vcoul[qpt_id, i] is an array corresponding to v(qpt_id + component[i]), or v(q+G)
    vcoul = f1['eps_header']['gspace']['vcoul']
    # matx: allows GG' matrix for qpoint qpt_id to be called as matx[qpt_id, freq]
    matx1 = f1['mats']['matrix'][:,0,:,:,:,0]
    matx2 = f2['mats']['matrix'][:,0,:,:,:,0]
    # handling complex matrix type:
    if f1['eps_header/flavor'][()] == 2:
        matx1 = matx1 + 1j*f1['mats']['matrix'][:,0,:,:,:,1]
        matx2 = matx2 + 1j*f2['mats']['matrix'][:,0,:,:,:,1]

    ###########################
    # creating and naming the output .h5 file
    eps_new = h5py.File(sys.argv[3],'w')
    
    # copying header from exact version if it exists
    if (len(sys.argv) == 5):
        headmat = h5py.File(sys.argv[4] , 'r')
    else:
        headmat = f1
    eps_new.copy(headmat['eps_header'],'eps_header')
    eps_new.copy(headmat['mf_header'],'mf_header')
    eps_new.copy(headmat['mats'],'mats')

    # loop over the q-points and write new superimposed eps^-1
    print("calculating eps^-1 from sublattice eps^-1's...\n")
    for i, qpt in enumerate(qpts):
        print(f"working on q-point #{i+1}, {qpt}")
        N_g = mat_sizes[i]
        N_freq = f1['eps_header/freqs/nfreq'][()]
        eps_q = matx_combine(matx1, matx2, i, N_g, vcoul, nfreq=N_freq)
        diag = np.diagonal(eps_q[0])
        eps_new['mats/matrix-diagonal'][i,:N_g,0] = diag.real
        #this should be almost identically zero:
        eps_new['mats/matrix-diagonal'][i,:N_g,1] = diag.imag

        # this is a loop over the frequencies associated w that qpoint inside eps_q
        for freq in range(N_freq):
            eps_new['mats/matrix'][i,0,freq,:N_g,:N_g,0] = eps_q[freq].real
            if f1['eps_header/flavor'][()] == 2:
                eps_new['mats/matrix'][i,0,freq,:N_g,:N_g,1] = eps_q[freq].imag
        print()
    eps_new.close()

    print("=="*10 + "\njob done")
    sys.stdout = old_stdout
    log_file.close()
    print("superposition finished")
    print()
