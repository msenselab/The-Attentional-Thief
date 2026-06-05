"""Generate the trial-triplet structure for the Attention Thief experiment."""

import numpy as np


def generate_triplets(seed=None):
    """Generate 6 randomized trial triplets.

    Design: 3 pic counts (8, 13, 18) with 2 reps each.
    Each triplet: active + passive + geometry (counterbalanced order for
    passive/geometry).

    Returns a list of 6 triplet dicts, each containing:
        - triplet_id: int (1-6)
        - n_pictures: int (8, 13, or 18)
        - repetition: int (1 or 2)
        - blocks: list of 3 block dicts with 'type' key
    """
    rng = np.random.default_rng(seed)

    # 2 reps each for 8, 13, and 18 → 6 triplets × 3 blocks = 18 blocks
    cells = [(8, 1), (8, 2), (13, 1), (13, 2), (18, 1), (18, 2)]

    triplets = []
    for i, (count, rep) in enumerate(cells):
        # Counterbalance passive/geometry order
        passive_first = ((i + rep) % 2 == 1)
        if passive_first:
            block_order = ['active', 'passive', 'geometry']
        else:
            block_order = ['active', 'geometry', 'passive']

        triplets.append({
            'triplet_id': i + 1,
            'n_pictures': count,
            'repetition': rep,
            'blocks': [{'type': bt} for bt in block_order],
        })

    rng.shuffle(triplets)

    return triplets


def sample_images_for_triplet(image_pool, n_pictures, seed=None):
    """Sample 2xN different images from pool (one set for active, one for passive).

    Returns a list of 2 lists, each containing N different images.
    No image appears in more than one set within a triplet.
    The geometry block does not require images.
    """
    needed = 2 * n_pictures
    if len(image_pool) < needed:
        raise ValueError(
            f"Image pool ({len(image_pool)}) too small for "
            f"2 × {n_pictures} = {needed} images"
        )

    rng = np.random.default_rng(seed)

    indices = rng.choice(len(image_pool), size=needed, replace=False)
    selected = [image_pool[i] for i in indices]

    sets = []
    for i in range(2):
        block_set = selected[i * n_pictures : (i + 1) * n_pictures]
        rng.shuffle(block_set)
        sets.append(list(block_set))

    return sets
