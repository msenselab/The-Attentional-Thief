"""Session-based experiment structure generation."""

import numpy as np

from .config import CONFIG


def generate_sessions(seed=None):
    """Generate session-based experiment structure.

    Design: 4 reps x 3 sessions (active, passive, constant).
    Within each session, 3 blocks with pic_counts [9, 12, 18] shuffled.
    Active session always runs first in each rep.
    Passive/constant order counterbalanced: odd reps vs even reps.

    Returns a list of 12 session dicts, each containing:
        - rep: int (1–4)
        - session_type: str ('active', 'passive', 'constant')
        - session_order: int (1–12, global ordering)
        - blocks: list of 3 block dicts with 'n_pictures' and 'block_order'
    """
    rng = np.random.default_rng(seed)
    pic_counts = list(CONFIG['pic_counts'])

    passive_first_odd = bool(rng.integers(2))

    sessions = []
    global_order = 0

    for rep in [1, 2, 3, 4]:
        active_counts = pic_counts.copy()
        rng.shuffle(active_counts)
        global_order += 1
        sessions.append({
            'rep': rep,
            'session_type': 'active',
            'session_order': global_order,
            'blocks': [{'n_pictures': n, 'block_order': i + 1}
                       for i, n in enumerate(active_counts)],
        })

        # Alternate passive/constant order across reps
        if rep % 2 == 1:
            second = 'passive' if passive_first_odd else 'constant'
            third = 'constant' if passive_first_odd else 'passive'
        else:
            second = 'constant' if passive_first_odd else 'passive'
            third = 'passive' if passive_first_odd else 'constant'

        for session_type in [second, third]:
            counts = pic_counts.copy()
            rng.shuffle(counts)
            global_order += 1
            sessions.append({
                'rep': rep,
                'session_type': session_type,
                'session_order': global_order,
                'blocks': [{'n_pictures': n, 'block_order': i + 1}
                           for i, n in enumerate(counts)],
            })

    return sessions


def allocate_images(pool, sessions, practice_n, seed=None):
    """Pre-allocate all images with 3-category balancing and global uniqueness.

    pool must be a dict with 'human', 'social', 'nature' keys (no replacement).

    Images required (4 reps × 3 session types × 3 blocks):
      Active:   4 × (9+12+18) = 156  images
      Passive:  4 × (9+12+18) = 156  images
      Constant: 4 × 3         =  12  images (1 per block)
      Practice:                    3  images
      Foils:    18 (half of 36 blocks get new probe, 6 per category)
      Total:                     345  images → ~115 per category

    Returns:
        practice_images : list of practice_n image paths
        session_images  : dict (rep, session_type, n_pictures) -> list of paths
        foil_images     : dict (rep, session_type, n_pictures) -> [1 foil path]
    """
    if not (isinstance(pool, dict) and
            'human' in pool and 'social' in pool and 'nature' in pool):
        raise ValueError("pool must have 'human', 'social', 'nature' keys")

    rng = np.random.default_rng(seed)
    return _allocate_balanced(pool, sessions, practice_n, rng)


def _allocate_balanced(pool, sessions, practice_n, rng):
    """Allocate images: equal human/social/nature balance, no global repeats."""
    human = list(pool['human'])
    social = list(pool['social'])
    nature = list(pool['nature'])
    rng.shuffle(human)
    rng.shuffle(social)
    rng.shuffle(nature)

    # Mutable cursors per category
    cur = [0, 0, 0]   # [human, social, nature]
    pools = [human, social, nature]
    names = ['human', 'social', 'nature']

    def get_n(n):
        """Take n images split equally across 3 categories."""
        n_each = n // 3
        remainder = n % 3  # 0, 1, or 2 — nature absorbs extras
        amounts = [n_each, n_each, n_each + remainder]
        for i, amt in enumerate(amounts):
            if cur[i] + amt > len(pools[i]):
                raise RuntimeError(
                    f"Not enough {names[i]} images "
                    f"(need {cur[i] + amt}, have {len(pools[i])})"
                )
        imgs = []
        for i, amt in enumerate(amounts):
            imgs.extend(pools[i][cur[i]:cur[i] + amt])
            cur[i] += amt
        rng.shuffle(imgs)
        return imgs

    constant_cat_order = []  # shuffled category indices, refilled every 3 calls

    def get_one():
        """Take 1 image per category, randomised within each round of 3."""
        if not constant_cat_order:
            order = [0, 1, 2]
            rng.shuffle(order)
            constant_cat_order.extend(order)
        idx = constant_cat_order.pop(0)
        if cur[idx] >= len(pools[idx]):
            raise RuntimeError(f"Not enough {names[idx]} images")
        img = pools[idx][cur[idx]]
        cur[idx] += 1
        return [img]

    # Practice
    practice_images = get_n(practice_n)

    # Block images
    session_images = {}
    for session in sessions:
        for block in session['blocks']:
            n = block['n_pictures']
            key = (session['rep'], session['session_type'], n)
            if session['session_type'] == 'constant':
                session_images[key] = get_one()
            else:
                session_images[key] = get_n(n)

    # Foil images: 18 total (6 per category), used for "new" probes in recognition.
    # Half of 36 blocks (18) will show a new probe; which blocks are determined at runtime.
    foil_images = get_n(18)  # balanced: 6 human + 6 social + 6 nature

    total_used = sum(cur)
    print(f"Images allocated: {cur[0]} human + {cur[1]} social + {cur[2]} nature = {total_used} total")

    return practice_images, session_images, foil_images
