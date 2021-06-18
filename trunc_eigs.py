#!/usr/bin/env python

# written by Jack McArthur, last edited 1/9/2021
# supervised by Prof. Diana Qiu
# inspiration from M Del Ben et al., Physical Review B 99, 125128 (2019)

# this program takes an inverse epsilon matrix, .h5 format, and writes an inverse epsilon matrix 
# simplified by the static subspace approximation (SSA) in the same plane wave basis as the original

import numpy as np
import h5py

def matx_truncate(matx, qpt, N_g, vcoul, eig_thresh=0.0, nfreq=1):
    # does SSA on a rank 3 matrix for a specific q-point with a given eigenvalue threshhold.
    
    # matx is the N_qpt x N_freq x N_G x N_G matrix from epsmat.h5 of eps^-1_GG' for all qpoints sampled
    # qpt is the index of the qpt in question
    # N_g comes from f['eps_header']['gspace']['nmtx']
    # vcoul comes from f['eps_header']['gspace']['vcoul']
    # eig_thresh is the eigenvalue threshhold for SSA (if negative, it is the frac. of eigenvalues to keep)
    # nfreq is the number of frequencies to do SSA on for each qpoint; defaults to all of them
    # eventual output, eps_inv is a list of length n_freq of N_G x N_G matrices for the qpoint in question
    # eps_inv[n] is the N_G x N_G matrix for the nth frequency at qpoint qpt
    eps_inv = []

    # helpful later
    v_G = np.diag(vcoul[qpt, :N_g])
    v_G_inv = np.diag(1. / vcoul[qpt, :N_g])
    v_G_sqrt = np.diag(np.sqrt(vcoul[qpt, :N_g]))
    v_G_sqrt_inv = np.diag(1. / np.sqrt(vcoul[qpt, :N_g]))

    # calculate chi0 from eps, from Del Ben equation 5
    # 0 in matx[i,0] is the ID of the relevant frequency
    # we transpose matx[i,0] because del Ben and BGW use different conventions
    chi0_GG = v_G_inv @ (-np.linalg.inv(matx[qpt,0][:N_g,:N_g].conj().T) + np.identity(N_g))
    #############################
    # working with symmetrized matrix now

    # we now calculate the symmetrized matrix form of v*chi, chi0_bar, which has same eigfuncs. as epsilon.
    # Del Ben equation 9
    chi0_bar = v_G_sqrt @ chi0_GG @ v_G_sqrt

    # decompose this symmetrized v*chi: these are the eigenvals used to truncate basis
    # argsort used to sort eigvals by size, making it easy to truncate them: Del Ben eqn 10
    lambda_q, C_q = np.linalg.eigh(-chi0_bar)
    idx = lambda_q.argsort()[::-1]
    lambda_q = lambda_q[idx]
    C_q = C_q[:,idx]

    # get truncated eigenbasis for eps (since -vX0 has same eigvals as eps and eps^-1)
    if eig_thresh > lambda_q[-1]:
        trunc_num = np.argmax(lambda_q < eig_thresh)
        print(f'keeping largest {trunc_num} of {len(C_q)} eigenvectors')
        Cs_q = C_q[:,:np.argmax(lambda_q < eig_thresh)]
    # if eig_thresh negative, truncate by fraction of eigenvectors to keep rather than threshhold
    elif 0 > eig_thresh >= -1:
        trunc_num = int(-eig_thresh*len(C_q))
        print(f'keeping the {trunc_num} of {len(C_q)} eigenvectors with eigvals. > {lambda_q[trunc_num]:.2}')
        Cs_q = C_q[:,:trunc_num]
    else:
        # if eig_thresh is 0 or otherwise smaller than smallest eigenvalue of vX0, don't truncate
        print(f'keeping all {len(C_q)} of {len(C_q)} eigenvectors')
        Cs_q = C_q

    # remember N_g = mat_size, this is size of the subspace
    N_b = len(Cs_q[0])

    # we've constructed the subspace with frequency omega = 0
    # now we project matrices for all the frequencies, 0 included, onto the subspace
    for freq in range(nfreq):
        if freq != 0:
            # combination of Del Ben eqns.5 and 9
            chi0_bar = v_G_sqrt_inv @ (-np.linalg.inv(matx[qpt,freq][:N_g,:N_g].conj().T) + np.identity(N_g)) @ v_G_sqrt
            
        # projecting chi onto truncated basis: Del Ben eqns 11 and 12
        chi0s_bar = Cs_q.conj().T @ chi0_bar @ Cs_q

        # now obtain eps_s_bar in static eigenvector basis, in text of Del Ben after eqn 12
        eps_s_bar = np.identity(N_b) - chi0s_bar
        eps_inv_s_bar = np.linalg.inv(eps_s_bar)
        # invert eps_s numerically, transform back to plane wave basis
        eps_inv_s = Cs_q @ (eps_inv_s_bar - np.identity(N_b)) @ Cs_q.conj().T + np.identity(N_g)

        # finally, eps_inv unsymmetrized is obtained
        # Del Ben in text after equation 12
        # (note that eps_inv is transposed back now)
        eps_inv.append((v_G_sqrt @ eps_inv_s @ v_G_sqrt_inv).conj().T)
    
    return eps_inv

if __name__=='__main__':

    import sys
    
    if len(sys.argv) != 3:
        print("Usage: trunc-eigs.py filename.h5 eig_thresh")
        print("eig_thresh between 0 and -1 interpreted as fraction of eigenvectors to retain")
        sys.exit()
    
    eig_thresh = float(sys.argv[2])
    
    if eig_thresh > 0:
        print("interpreting eig_thresh as a threshhold")
        print(f"writing to log file trunc-eigs-{eig_thresh}.out")
        log_file = open(f"trunc-eigs-{eig_thresh}.out","w")
    
    elif eig_thresh == 0.00:
        print("not truncating: will return an output identical to the input")
        print(f"writing to log file trunc-eigs-{eig_thresh}.out")
        log_file = open(f"trunc-eigs-{eig_thresh}.out","w")
        
    elif eig_thresh > -1:
        print("interpreting eig_thresh as a fraction of eigenvalues to keep")
        print(f"writing to log file trunc-eigs-frac-{-eig_thresh}.out")
        log_file = open(f"trunc-eigs-frac-{-eig_thresh}.out","w")
    else:
        print("unacceptable eig_thresh:")
        print("eigh_thresh > 0 interpreted as threshhold, -1 < eig_thresh < 0 interpreted as fraction")
        sys.exit()
    
    old_stdout = sys.stdout
    sys.stdout = log_file

    # Open old aqnd new epsmat files and copy header info
    print(f"opening {sys.argv[1]}...\n")
    f_in = sys.argv[1]
    f = h5py.File(f_in , 'r')
    
    print(f"there are {f['eps_header/qpoints/nq'][()]} q-points with matrices for {f['eps_header/freqs/nfreq'][()]} frequencies each")
    
    # qpts: list of qpts stored in f['mats']['matrix']
    qpts = f['eps_header']['qpoints']['qpts']
    # matx: allows GG' matrix for qpoint qpt_id to be called as matx[qpt_id, freq]
    matx = f['mats']['matrix'][:,0,:,:,:,0]
    # handling complex matrix type:
    if f['eps_header/flavor'][()] == 2:
        matx = matx + 1j*f['mats']['matrix'][:,0,:,:,:,1]
    # mat_size: mat_size[qpt_id] gives size of matx[qpt_id], as all eps^-1_GG'(q) 
    # have the same shape (n x n), but some have empty rows at the end
    mat_sizes = f['eps_header']['gspace']['nmtx']
    # vcoul: vcoul[qpt_id, i] is an array corresponding to v(qpt_id + component[i]), or v(q+G)
    vcoul = f['eps_header']['gspace']['vcoul']
    
    ###########################
    # creating and naming the output .h5 file
    if eig_thresh >= 0:
        eps_new  = h5py.File(f'{f_in[:-3]}_trunc-{eig_thresh}.h5','w')
    else:
        eps_new = h5py.File(f'{f_in[:-3]}_trunc-frac-{-eig_thresh}.h5','w')

    eps_new.copy(f['eps_header'],'eps_header')
    eps_new.copy(f['mf_header'],'mf_header')
    eps_new.copy(f['mats'],'mats')

    # loop over the q-points and write new truncated eps^-1 with matx_truncate
    print("calculating eps^-1 in truncated basis...\n")
    for i, qpt in enumerate(qpts):
        print(f"working on q-point #{i+1}, {qpt}")
        N_g = mat_sizes[i]
        N_freq = f['eps_header/freqs/nfreq'][()]
        eps_q = matx_truncate(matx, i, N_g, vcoul, eig_thresh, N_freq)
        diag = np.diagonal(eps_q[0])
        eps_new['mats/matrix-diagonal'][i,:N_g,0] = diag.real
        #this should be almost identically zero:
        eps_new['mats/matrix-diagonal'][i,:N_g,1] = diag.imag
        
        # this is a loop over the frequencies associated w that kpoint inside eps_q
        for freq in range(N_freq):
            eps_new['mats/matrix'][i,0,freq,:N_g,:N_g,0] = eps_q[freq].real
            if f['eps_header/flavor'][()] == 2:
                eps_new['mats/matrix'][i,0,freq,:N_g,:N_g,1] = eps_q[freq].imag
        print()
    eps_new.close()
    
    print("=="*10 + "\njob done")
    sys.stdout = old_stdout
    log_file.close()
    print("truncation finished")
    print()
    