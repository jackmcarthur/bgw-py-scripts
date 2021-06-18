#!/usr/bin/env python

# written by Jack McArthur, last edited 2/16/2021
# supervised by Prof. Diana Qiu

# this program calculates chi0^-1(r,r') from eps^-1(q,G,G'), spread out over
# two files, eps0mat.h5 and epsmat.h5. the rank three result is written to another h5.
# this version is parallelized with mpi4py

import numpy as np
import math
import h5py
from mpi4py import MPI

# helper function for calculating the phase factors associated with the 
# fourier transform of eps^-1(q,G,G').
def lr_phase(r1, r2, components):
    Gdotr1 = np.dot(components, r1)
    Gdotr2 = np.dot(components, r2)
    return np.exp(np.pi * 2j * np.subtract.outer(Gdotr1, Gdotr2))

# this allows one to find eps between two points
def eps_rr_pts(r, r_p, components, matx):
    lrphase = lr_phase(r_p, r, components)
    result = 0j
    for i in range(matx.shape[0]):
        result += np.sum(lrphase * matx[i])
    return result

# this allows one to find eps with r_px, r_py, r_pz as part of an mgrid
def eps_rr_grid(r, r_px, r_py, r_pz, components, matx):
    lrphase = lr_phase(r, np.array([r_px.flatten(), r_py.flatten(), r_pz.flatten()]), components)
    result = np.zeros(r_px.flatten().shape, dtype=np.complex128)
    # loop over qpts
    for jj in range(lrphase.shape[2]):
        for i in range(matx.shape[0]):
            result[jj] += np.sum(lrphase[:,:,jj] * matx[i])
    return result.reshape(r_px.shape)
    
# static epsmat class, constructed from an HDF5 file
class Epsmat:
    def __init__(self, file):
        self.f = h5py.File(file, 'r')
        
        # necessary for math
        self.components = self.f['mf_header/gspace/components']
        self.qpts = self.f['eps_header/qpoints/qpts']
        self.matx = self.f['mats/matrix'][:,0,0,:,:,0]
        self.mat_sizes = self.f['eps_header/gspace/nmtx']
        self.vcoul = self.f['eps_header/gspace/vcoul']
        
        if self.f['eps_header/flavor'][()] == 2:
            self.matx = self.matx + 1j*self.f['mats/matrix'][:,0,0,:,:,1]
            
        # get static chi0
        self.nonintchi = np.zeros_like(self.matx)
        for qpt in range(self.nonintchi.shape[0]):
            N_g = self.mat_sizes[qpt]
            self.nonintchi[qpt,:N_g,:N_g] = self.get_nonintchi(qpt)
            
        # this is to avoid changing the eps_rr code
        self.matx = self.nonintchi
        
        #necessary for plotting
        self.coords = self.f['mf_header/crystal/apos'][:]
        self.atom_ids = self.f['mf_header/crystal/atyp'][:]
        self.cell = self.f['mf_header/crystal/avec'][:]
        
    def get_nonintchi(self, qpt, freq=0):
        N_g = self.mat_sizes[qpt]
        v_G_inv = np.diag(1. / self.vcoul[qpt, :N_g])
        return v_G_inv @ (-np.linalg.inv(self.matx[qpt][:N_g,:N_g].conj().T) + np.identity(N_g))

if __name__=='__main__':

    import sys

    if (len(sys.argv) != 6):
        print("Usage: get_chi_rr.py epsmat.h5 eps0mat.h5 n_X n_Y n_Z")
        sys.exit()

    log_file = open(f'chi_rr.log',"w")
    old_stdout = sys.stdout
    sys.stdout = log_file

    # Open both epsmat files
    print(f"opening {sys.argv[1]}...\n")
    eps0 = Epsmat(sys.argv[1])
    nq = eps0.f['eps_header/qpoints/nq'][()]
    nfreq = eps0.f['eps_header/freqs/nfreq'][()]
    print(f"there are {nq} q-points with matrices for {nfreq} frequencies each\n")
    
    print(f"opening {sys.argv[2]}...\n")
    eps = Epsmat(sys.argv[2])
    nq = eps.f['eps_header/qpoints/nq'][()]
    nfreq = eps.f['eps_header/freqs/nfreq'][()]
    print(f"there are {nq} q-points with matrices for {nfreq} frequencies each\n")

    # qpts: list of qpts stored in f['mats']['matrix']
    qpts = np.concatenate((eps0.qpts, eps.qpts))

    # mat_size: mat_size[qpt_id] gives size of matx[qpt_id], as all eps^-1_GG'(q) 
    # have the same shape (n x n), but some have empty rows at the end
    mat_sizes = np.concatenate((eps0.mat_sizes, eps.mat_sizes))

    # matx: allows matrix for qpoint qpt_id to be called as matx[qpt_id, freq] 
    # we are choosing frequency = 0 here, to plot the static dielectric matrix
    matx0pad = np.zeros_like(eps.matx[0])
    matx0pad[:eps0.matx.shape[1],:eps0.matx.shape[2]] = eps0.matx[0]
    matx = np.concatenate((np.array([matx0pad]), eps.matx))

    # vcoul: vcoul[qpt_id, i] is an array corresponding to v(qpt_id + component[i]), or v(q+G)
    vcoul0 = np.zeros((1,1376))
    vcoul0[0,:eps0.mat_sizes[0]] = eps0.vcoul[:eps0.mat_sizes[0]]
    vcoul = np.concatenate((vcoul0, eps.vcoul))

    # components: g-space vectors for each matrix row/col
    components = np.asarray(eps.components)[:np.max(eps.mat_sizes)]

    ###########################
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    # dimensions of the grid that we calculate values for
    d = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    # reference point, r in eps^-1(r, r') as r' varies over unit cell
    r = (0., 0., 0.)
    
    epsset = np.zeros(d, dtype=np.complex128)
    # next four lines are weird to avoid doing redundant calc on opp sides of unit cell
    X, Y, Z  = np.mgrid[0:1:(d[0]+1)*1j , 0:1:(d[1]+1)*1j , 0:1:(d[2]+1)*1j]
    X = X[:d[0],:d[1],:d[2]]
    Y = Y[:d[0],:d[1],:d[2]]
    Z = Z[:d[0],:d[1],:d[2]]
    cell = eps0.cell
    
    # the commented out portion of this line can be useful for a crystal that does not have
    # orthorhombic symmetry or higher; it allows one to get the grid for a set of points corresponding to
    # the unit cube in cartesian coordinates.
    Xnew, Ynew, Znew = np.array([X.flatten(), Y.flatten(), Z.flatten()])#.T.dot(np.linalg.inv(cell).T).T
    
    Xnew = Xnew.reshape(d)
    Ynew = Ynew.reshape(d)
    Znew = Znew.reshape(d)

    # to reduce memory usage, we calculate epsset one 1D array at a time.
    # we use MPI to split this operation into chunks (along X), since every r' is independent.
    
    m = int(math.ceil(float(d[0]) / size))
    X_chunk = Xnew[rank*m:(rank+1)*m,:,:]
    Y_chunk = Ynew[rank*m:(rank+1)*m,:,:]
    Z_chunk = Znew[rank*m:(rank+1)*m,:,:]
    print(f'my m is {m}')
    
    epsset_chunk = np.zeros(X_chunk.shape, dtype=np.complex128)
    
    # this is the X dimension of the real space r-primes, over which we parallelize
    for i in range(m):
        print(f'working on x = {Xnew[i,0,0]}')
        # this is the Y dimension
        for j in range(d[1]):
            epsset_chunk[i,j] = eps_rr_grid(r, X_chunk[i,j], Y_chunk[i,j], Z_chunk[i,j], components, matx)
    print(epsset_chunk.shape)
    
    epsout = None
    if rank == 0:
      epsout = np.empty(d, dtype=np.complex128)
    comm.Gather(epsset_chunk, epsout, root=0)
    
    if rank == 0:
      print("now writing output to chi_rr_out.h5 ...")
      h5f = h5py.File('chi_rr_out.h5', 'w')
      h5f.create_dataset('dataset_1', data=epsout)
      h5f.close()
      
      print("=="*10 + "\njob done")
      sys.stdout = old_stdout
      log_file.close()
      print("eps^-1(r,r') finished")
      print()