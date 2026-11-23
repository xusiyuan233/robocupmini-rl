"""Contact geometry checks and source-driven rigid-sole clearance projection.

A BVH ankle/toe is not the sole of this robot. Never use max(mesh_z, 0) as a
swing correction: that can turn most of a swing into a foot sliding on the floor.
These functions neither certify dynamics nor implement AMP.
"""
from __future__ import annotations

import numpy as np


def intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def interior_mask(mask: np.ndarray, guard_frames: int) -> np.ndarray:
    """Exclude contact-transition samples and both clip edges; never wrap a clip."""
    if guard_frames < 0:
        raise ValueError('guard_frames must be nonnegative')
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a [frames, feet] mask')
    result = np.zeros_like(mask)
    for side in range(mask.shape[1]):
        for start, stop in intervals(mask[:, side]):
            if stop - start > 2 * guard_frames:
                result[start + guard_frames:stop - guard_frames, side] = True
    return result


def source_swing_clearance(times: np.ndarray, ankle_height: np.ndarray,
                           contact: np.ndarray, spatial_scale: float,
                           blend_s: float = .24,
                           minimum_mid_swing_m: float = .015) -> np.ndarray:
    """Project measured ankle lift to a rigid sole, without inventing gait cycles.

    The stance median provides the bind height. Source lift and source contact
    intervals are retained. A smooth 15 mm interior floor prevents a rigid shoe
    from scuffing during low human toe/heel clearance. This is a documented
    morphology adaptation, NOT a force-measured contact model.
    """
    times = np.asarray(times, dtype=float)
    ankle_height = np.asarray(ankle_height, dtype=float)
    contact = np.asarray(contact, dtype=bool)
    if (times.ndim != 1 or len(times) < 3 or
            ankle_height.shape != times.shape or contact.shape != times.shape or
            not np.isfinite(times).all() or not np.isfinite(ankle_height).all() or
            np.any(np.diff(times) <= 0) or not contact.any() or
            not np.isfinite(spatial_scale) or spatial_scale <= 0 or
            not np.isfinite(blend_s) or blend_s <= 0 or
            not np.isfinite(minimum_mid_swing_m) or minimum_mid_swing_m <= 0):
        raise ValueError('Invalid source/contact/clearance parameters')
    baseline = float(np.median(ankle_height[contact]))
    lift = np.maximum((ankle_height - baseline) * spatial_scale, 0.)
    # Smooth maximum rather than a derivative kink where measured lift crosses
    # the robot-specific minimum clearance.
    minimum = minimum_mid_swing_m
    safe_lift = .5 * (lift + minimum + np.sqrt((lift - minimum)**2 + .001**2))
    result = np.zeros_like(lift)

    def smoothstep(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0., 1.)
        return value**3 * (10. - 15. * value + 6. * value**2)

    for start, stop in intervals(~contact):
        weight = np.ones(stop - start)
        if start > 0:
            weight *= smoothstep((times[start:stop] - times[start]) / blend_s)
        if stop < len(times):
            weight *= smoothstep((times[stop-1] - times[start:stop]) / blend_s)
        result[start:stop] = safe_lift[start:stop] * weight
    return result


def contact_geometry_report(mesh_min_z: np.ndarray, anchor_velocity: np.ndarray,
                            contact: np.ndarray, dt: float) -> dict:
    """Test the actual transformed collision mesh, not the ankle/body origin."""
    mesh_min_z = np.asarray(mesh_min_z, dtype=float)
    velocity = np.asarray(anchor_velocity, dtype=float)
    contact = np.asarray(contact, dtype=bool)
    if (mesh_min_z.shape != contact.shape or contact.ndim != 2 or
            velocity.shape != contact.shape + (3,) or
            not np.isfinite(mesh_min_z).all() or not np.isfinite(velocity).all() or
            not np.isfinite(dt) or dt <= 0):
        raise ValueError('Invalid foot geometry arrays')
    # 100 ms guard excludes heel/toe/contact transition ambiguities, but not
    # whole swings. Excluding clip endpoints also avoids invented loop contact.
    guard = max(1, int(np.ceil(.10 / dt)))
    swing = interior_mask(~contact, guard)
    stance = interior_mask(contact, guard)
    horizontal_speed = np.linalg.norm(velocity[..., :2], axis=-1)
    swing_scrape = swing & (mesh_min_z < .003)
    dragging = swing_scrape & (horizontal_speed > .02)
    stance_sliding = stance & (horizontal_speed > .01)
    return {
        'transition_guard_frames': guard,
        'minimum_swing_clearance_threshold_m': .003,
        'swing_interior_samples': int(swing.sum()),
        'swing_scrape_samples': int(swing_scrape.sum()),
        'moving_swing_ground_samples': int(dragging.sum()),
        'swing_scrape_fraction': float(swing_scrape.sum() / max(swing.sum(), 1)),
        'minimum_interior_swing_mesh_z_m': float(mesh_min_z[swing].min()) if swing.any() else None,
        'stance_interior_samples': int(stance.sum()),
        'stance_sliding_samples': int(stance_sliding.sum()),
        'maximum_stance_horizontal_speed_m_s': float(horizontal_speed[stance].max()) if stance.any() else None,
        'mesh_below_ground_2mm_samples': int((mesh_min_z < -.002).sum()),
        'passed': bool(swing.any() and stance.any() and not swing_scrape.any()
                       and not stance_sliding.any() and not (mesh_min_z < -.002).any()),
        'note': 'Necessary geometry/contact consistency only; not a dynamics or AMP certificate.',
    }
