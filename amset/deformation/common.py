import numpy as np

_voigt_idxs = ([0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1])
_fmt_str = "[{0:{space}.{dp}f} {1:{space}.{dp}f} {2:{space}.{dp}f} {3:{space}.{dp}f} {4:{space}.{dp}f} {5:{space}.{dp}f}]"


def get_formatted_tensors(tensors):
    min_n = np.inf
    for tensor in tensors:
        tensor = tensor.round(5)
        frac_t = tensor - np.round(tensor)
        min_n = min(np.min(np.abs(frac_t[np.nonzero(frac_t)])), min_n)

    # get the number of decimal places we will need
    dp = np.ceil(np.log10(1 / min_n)).astype(int)
    space = dp + 3

    formatted_tensors = []
    for tensor in tensors:
        voigt = np.array(tensor)[_voigt_idxs].round(5)
        voigt[voigt == -0.0] = 0
        formatted_tensors.append(_fmt_str.format(*voigt, space=space, dp=dp))

    return formatted_tensors
