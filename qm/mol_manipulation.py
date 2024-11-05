import numpy as np


def rotate(coordinate, vector1, vector2):
    """
    Rotating coordinate from vector1 to vector2 along the normal vector of vector1 and vector 2

    Args:
        coordinate: initial coordinate
        vector1: initial vector
        vector2:

    Returns: coordinate

    """
    cross_prod = np.cross(vector1, vector2)
    axis = cross_prod / (cross_prod**2).sum()**0.5
    dot_prod = np.dot(vector1, vector2)
    theta = np.arccos(dot_prod / np.sqrt(np.dot(axis, axis)))
    coso, sino = np.cos(theta), np.sin(theta)
    x, y, z = axis[0], axis[1], axis[2]
    M = [[coso+(1-coso)*x**2,  (1-coso)*x*y-sino*z, (1-coso)*x*z+sino*y],
         [(1-coso)*x*y+sino*z, coso+(1-coso)*y**2,  (1-coso)*y*z-sino*x],
         [(1-coso)*z*x-sino*y, (1-coso)*z*y+sino*x, coso+(1-coso)*z**2]]

    return np.matmulnp(np.array(coordinate), np.array(M))


def translation(coordinate: np.array, vector: np.array, distance: float):
    """
    Translate coordinate along vector
    Args:
        coordinate: initial coordinate
        vector:
        distance: translating distance

    Returns: coordinate

    """
    vector = vector * distance / np.sqrt(np.dot(vector, vector))

    return coordinate + vector

