import numpy as np
from glp import periodic

def ternary(x, y, z):
    """Linearise replica indices."""
    if x == -1:
        x = 2
    if y == -1:
        y = 2
    if z == -1:
        z = 2

    return z * 9 + y * 3 + x

# big clown emoji. but it works!
offsets = np.zeros((27, 3))

for x in [-1, 0, 1]:
    for y in [-1, 0, 1]:
        for z in [-1, 0, 1]:
            offsets[ternary(x, y, z)] = [x, y, z]


def _np_unfolding(positions, cell, cutoff):
    """Get unfolding instructions.

    The idea is very simple: We first compute whether any given atom lies within
    a distance of `cutoff` to any of the 6 unit cell boundaries, i.e. we do a kind
    of collision detection with these regions. Then, we simply go through the 26 different
    ways an atom might have to be replicated, in three different types of replicas:

    (1) On the six faces of the unit cell
    (2) Along the 12 edges, and
    (3) Along the 8 corners.

    (In total, together with the original unit cell, we have the 3x3x3=27 possible unit
    cell replicas in which we may have to place an atom. (In the worst case, it'll of
    course be only 1 corner, 3 faces, 3 edges, so 7 replicas.))

    The rules for this are simple:
    - If it collides with a face, place a replica in face at the opposite face,
    - if it collides with two faces, add a replica at the opposite edge,
    - if it collides with three faces, add a replica at the opposite corner.

    The advantage of all of this is that this can be fully vectorised:
    It's just boolean checks over arrays, which is very fast.

    In order to treat non-orthorhombic cells, we need to measure the distance from the
    face of the parallelpepiped, which we achieve by projecting onto surface normals.
    (Alternatively, we could do this in scaled coordinates.)

    """

    heights = periodic.get_heights(cell)
    projections = periodic.project_on_normals(cell, positions)

    # projections = project_onto_planes(positions, cell)

    for length in heights:
        assert length > 0  # ensure that normals point into cell
        assert cutoff < length  # we can only duplicate once in each direction

    projections_x = projections[:, 0]
    projections_y = projections[:, 1]
    projections_z = projections[:, 2]

    # which atom is within the cutoff of which face?
    collisions = np.zeros((positions.shape[0], 6), dtype=bool)

    collisions[:, 0] = projections_x <= cutoff
    collisions[:, 1] = projections_x >= heights[0] - cutoff

    collisions[:, 2] = projections_y <= cutoff
    collisions[:, 3] = projections_y >= heights[1] - cutoff

    collisions[:, 4] = projections_z <= cutoff
    collisions[:, 5] = projections_z >= heights[2] - cutoff

    # filter down to atoms that touch a border, the rest we don't need to think about
    candidates = np.argwhere(np.any(collisions, axis=1)).flatten()
    collisions = collisions[candidates]

    # for each candidate, we'll put True in for every replica that we want to place it in
    replicas = np.zeros((candidates.shape[0], 27), dtype=bool)
    # the indexing is in ternary along x, y, z
    # i.e. (0, 0, 0) = 0, (0, 0, 1) = 1, (0, 0, -1) = 2

    # easier to read
    x_lo = collisions[:, 0]  # collided with face in -x direction
    x_hi = collisions[:, 1]  # collided with face in +x direction
    y_lo = collisions[:, 2]  # ... etc ...
    y_hi = collisions[:, 3]
    z_lo = collisions[:, 4]
    z_hi = collisions[:, 5]

    # 6 faces

    replicas[:, ternary(+1, 0, 0)] = x_lo
    replicas[:, ternary(-1, 0, 0)] = x_hi

    replicas[:, ternary(0, +1, 0)] = y_lo
    replicas[:, ternary(0, -1, 0)] = y_hi

    replicas[:, ternary(0, 0, +1)] = z_lo
    replicas[:, ternary(0, 0, -1)] = z_hi

    # 12 edges

    replicas[:, ternary(+1, +1, 0)] = x_lo & y_lo
    replicas[:, ternary(+1, -1, 0)] = x_lo & y_hi

    replicas[:, ternary(-1, +1, 0)] = x_hi & y_lo
    replicas[:, ternary(-1, -1, 0)] = x_hi & y_hi

    replicas[:, ternary(0, +1, +1)] = y_lo & z_lo
    replicas[:, ternary(0, +1, -1)] = y_lo & z_hi

    replicas[:, ternary(0, -1, +1)] = y_hi & z_lo
    replicas[:, ternary(0, -1, -1)] = y_hi & z_hi

    replicas[:, ternary(+1, 0, +1)] = x_lo & z_lo
    replicas[:, ternary(+1, 0, -1)] = x_lo & z_hi

    replicas[:, ternary(-1, 0, +1)] = x_hi & z_lo
    replicas[:, ternary(-1, 0, -1)] = x_hi & z_hi

    # 8 corners

    replicas[:, ternary(+1, +1, +1)] = x_lo & y_lo & z_lo
    replicas[:, ternary(+1, +1, -1)] = x_lo & y_lo & z_hi

    replicas[:, ternary(-1, +1, -1)] = x_hi & y_lo & z_hi
    replicas[:, ternary(-1, +1, +1)] = x_hi & y_lo & z_lo

    replicas[:, ternary(+1, -1, +1)] = x_lo & y_hi & z_lo
    replicas[:, ternary(-1, -1, +1)] = x_hi & y_hi & z_lo

    replicas[:, ternary(-1, -1, -1)] = x_hi & y_hi & z_hi
    replicas[:, ternary(+1, -1, -1)] = x_lo & y_hi & z_hi

    idx_candidate_and_direction = np.argwhere(
        replicas
    )  # [collision index, type of replica]

    idx = candidates[
        idx_candidate_and_direction[:, 0]  # idx in the collisions array
    ]  # idx in the original positions array
    directions = idx_candidate_and_direction[:, 1]

    return idx, offsets[directions]


