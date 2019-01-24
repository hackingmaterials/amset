from monty.json import MSONable


class Valley(MSONable):

    def __init__(self, kpoints, kpoints_norm, velocities,
                 velocities_norm, a_contrib, c_contrib,
                 angle_k_prime_mapping=None):
        self.kpoints = kpoints
        self.kpoints_norm = kpoints_norm
        self.velocities = velocities
        self.velocities_norm = velocities_norm
        self.a_contrib = a_contrib
        self.c_contrib = c_contrib
        self.angle_k_prime_mapping = angle_k_prime_mapping

