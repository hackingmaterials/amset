import multiprocessing as mp

import numpy as np
from BoltzTraP2.fite import BOLTZMANN, FFTc, FFTev

from amset.constants import defaults
from amset.electronic_structure.fd import dfdde, fd


def get_bands_fft(
    equivalences,
    coeffs,
    lattvec,
    return_effective_mass=False,
    nworkers=defaults["nworkers"],
):
    """Rebuild the full energy bands from the interpolation coefficients.

    Args:
        equivalences: list of k-point equivalence classes in direct coordinates
        coeffs: interpolation coefficients
        lattvec: lattice vectors of the system
        return_effective_mass: Whether to calculate the effective mass.
        nworkers: number of working processes to span

    Returns:
        A 3-tuple (eband, vvband, cband): energy bands, v x v outer product
        of the velocities, and curvature of the bands (if requested). The
        shapes of those arrays are (nbands, nkpoints), (nbands, 3, 3, nkpoints)
        and (nbands, 3, 3, 3, nkpoints), where nkpoints is the total number of
        k points on the grid. If curvature is None, so will the third element
        of the tuple.
    """
    dallvec = np.vstack(equivalences)
    sallvec = mp.sharedctypes.RawArray("d", dallvec.shape[0] * 3)
    allvec = np.frombuffer(sallvec)
    allvec.shape = (-1, 3)
    dims = 2 * np.max(np.abs(dallvec), axis=0) + 1
    np.matmul(dallvec, lattvec.T, out=allvec)
    eband = np.zeros((len(coeffs), np.prod(dims)))
    vvband = np.zeros((len(coeffs), 3, 3, np.prod(dims)))
    vb = np.zeros((len(coeffs), 3, np.prod(dims)))
    if return_effective_mass:
        effective_mass = np.zeros((len(coeffs), 3, 3, np.prod(dims)))
    else:
        effective_mass = None

    # Span as many worker processes as needed, put all the bands in the queue,
    # and let them work until all the required FFTs have been computed.
    workers = []
    iqueue = mp.Queue()
    oqueue = mp.Queue()
    for iband, bandcoeff in enumerate(coeffs):
        iqueue.put((iband, bandcoeff))
    # The "None"s at the end of the queue signal the workers that there are
    # no more jobs left and they must therefore exit.
    for i in range(nworkers):
        iqueue.put(None)
    for i in range(nworkers):
        workers.append(
            mp.Process(
                target=fft_worker,
                args=(
                    equivalences,
                    sallvec,
                    dims,
                    iqueue,
                    oqueue,
                    return_effective_mass,
                ),
            )
        )
    for w in workers:
        w.start()
    # The results of the FFTs are processed as soon as they are ready.
    for r in range(len(coeffs)):
        iband, eband[iband], vvband[iband], cb, vb[iband] = oqueue.get()
        if return_effective_mass:
            effective_mass[iband] = cb
    for w in workers:
        w.join()
    if effective_mass is not None:
        effective_mass = effective_mass.real
    return eband.real, vvband.real, effective_mass, vb


def fft_worker(
    equivalences, sallvec, dims, iqueue, oqueue, return_effective_mass=False
):
    """Thin wrapper around FFTev and FFTc to be used as a worker function.

    Args:
        equivalences: list of k-point equivalence classes in direct coordinates
        sallvec: Cartesian coordinates of all k points as a 1D vector stored
                    in shared memory.
        dims: upper bound on the dimensions of the k-point grid
        iqueue: input multiprocessing.Queue used to read bad indices
            and coefficients.
        oqueue: output multiprocessing.Queue where all results of the
            interpolation are put. Each element of the queue is a 4-tuple
            of the form (index, eband, vvband, cband), containing the band
            index, the energies, the v x v outer product and the curvatures
            if requested.
        return_effective_mass: Whether to calculate the effective mass.

    Returns:
        None. The results of the calculation are put in oqueue.
    """
    iu0 = np.triu_indices(3)
    il1 = np.tril_indices(3, -1)
    iu1 = np.triu_indices(3, 1)
    allvec = np.frombuffer(sallvec)
    allvec.shape = (-1, 3)

    while True:
        task = iqueue.get()
        if task is None:
            break
        else:
            index, bandcoeff = task
        eband, vb = FFTev(equivalences, bandcoeff, allvec, dims)
        vvband = np.zeros((3, 3, np.prod(dims)))
        effective_mass = np.zeros((3, 3, np.prod(dims)))

        vvband[iu0[0], iu0[1]] = vb[iu0[0]] * vb[iu0[1]]
        vvband[il1[0], il1[1]] = vvband[iu1[0], iu1[1]]
        if return_effective_mass:
            effective_mass[iu0] = FFTc(equivalences, bandcoeff, allvec, dims)
            effective_mass[il1] = effective_mass[iu1]
            effective_mass = np.linalg.inv(effective_mass.T).T
        else:
            effective_mass = None
        oqueue.put((index, eband, vvband, effective_mass, vb))


def fermiintegrals(epsilon, dos, sigma, mur, Tr, dosweight=2.0, cdos=None):
    """Compute the moments of the FD distribution over the band structure.

    Args:
        epsilon: array of energies at which the DOS is available
        dos: density of states
        sigma: transport DOS
        mur: array of chemical potential values
        Tr: array of temperature values
        dosweight: maximum occupancy of an electron mode
        cdos: "curvature DOS" if available

    Returns:
        Five numpy arrays, namely:
        1. An (nT, nmu) array with the electron counts for each temperature and
           each chemical potential.
        2. An (nT, nmu, 3, 3) with the integrals of the 3 x 3 transport DOS
           over the band structure taking the occupancies into account.
        3. An (nT, nmu, 3, 3) with the first moment of the 3 x 3 transport DOS
           over the band structure taking the occupancies into account.
        4. An (nT, nmu, 3, 3) with the second moment of the 3 x 3 transport DOS
           over the band structure taking the occupancies into account.
        5. If the cdos argument is provided, an (nT, nmu, 3, 3, 3) with the
           integrals of the 3 x 3 x 3 "curvature DOS" over the band structure
           taking the occupancies into account.
        where nT and nmu are the sizes of Tr and mur, respectively.
    """
    kBTr = np.array(Tr) * BOLTZMANN
    nT = len(Tr)
    nmu = len(mur)
    N = np.empty((nT, nmu))
    L0 = np.empty((nT, nmu, 3, 3))
    L1 = np.empty((nT, nmu, 3, 3))
    L2 = np.empty((nT, nmu, 3, 3))
    if cdos is not None:
        L11 = np.empty((nT, nmu, 3, 3, 3))
    else:
        L11 = None
    de = epsilon[1] - epsilon[0]
    for iT, kBT in enumerate(kBTr):
        for imu, mu in enumerate(mur):
            N[iT, imu] = -(dosweight * dos * fd(epsilon, mu, kBT)).sum() * de
            int0 = -dosweight * dfdde(epsilon, mu, kBT)
            intn = int0 * sigma
            L0[iT, imu] = intn.sum(axis=2) * de
            intn *= epsilon - mu
            L1[iT, imu] = -intn.sum(axis=2) * de
            intn *= epsilon - mu
            L2[iT, imu] = intn.sum(axis=2) * de
            if cdos is not None:
                cint = int0 * cdos
                L11[iT, imu] = -cint.sum(axis=3) * de
    return N, L0, L1, L2, L11
