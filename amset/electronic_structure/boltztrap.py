import multiprocessing as mp

import numpy as np

from amset.constants import defaults
from BoltzTraP2.fite import FFTc, FFTev


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
