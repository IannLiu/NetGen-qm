import numpy as np


def load_coor_from_xyz_file(filename: str):
    coor = []
    with open(filename, "r") as xyz_file:
        n_atoms = int(xyz_file.readline().split()[0])

        # XYZ lines should be the following 2 + n_atoms lines
        xyz_lines = xyz_file.readlines()[1: n_atoms + 1]

        for i, line in enumerate(xyz_lines):
            atom_label, x, y, z = line.split()[:4]
            coor.append([float(x), float(y), float(z)])

        if len(coor) != n_atoms:
            raise ValueError(f"Inconsistent xyz file {filename}")

    return np.array(coor)