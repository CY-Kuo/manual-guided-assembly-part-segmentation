# smart_prior.py
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None


def _to_bool(x):
    return (x.astype(np.uint8) > 0)

def _resize_bool(mask, out_hw):
    H, W = out_hw
    if mask.shape == (H, W):
        return _to_bool(mask)
    if cv2 is not None:
        m = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return _to_bool(m)
    from PIL import Image
    m = np.array(Image.fromarray(mask.astype(np.uint8)*255).resize((W, H), Image.NEAREST)) > 127
    return m

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _distance_to_student(student_bool):
    """Distance from each pixel to the nearest student pixel.
    0 inside student; grows outside. Robust when student is empty."""
    s = _to_bool(student_bool)
    if s.sum() == 0:
        # No student coverage: return large uniform distance so everything is "far".
        return np.full(s.shape, 1e6, dtype=np.float32)

    # distance outside the student: distance_transform on inverse mask
    if ndi is not None:
        dist_out = ndi.distance_transform_edt(~s)
    elif cv2 is not None:
        dist_out = cv2.distanceTransform((~s).astype(np.uint8), cv2.DIST_L2, 3)
    else:
        # very slow fallback: taxicab distance iterative (only as a last resort)
        H, W = s.shape
        inf = H + W
        d = np.full((H, W), inf, dtype=np.int32)
        ys, xs = np.where(s)
        for y, x in zip(ys, xs):
            d = np.minimum(d, np.abs(np.arange(H)[:, None]-y) + np.abs(np.arange(W)[None, :]-x))
        dist_out = d.astype(np.float32)
    # inside student we want 0, and outside we use dist_out
    return dist_out

def _morphology(mask, erode=0, open_it=0, close_it=0):
    m = mask.copy().astype(np.uint8)
    if cv2 is not None:
        # 3x3 kernel
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for _ in range(int(erode)):
            m = cv2.erode(m, k, iterations=1)
        for _ in range(int(open_it)):
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        for _ in range(int(close_it)):
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
        return m.astype(bool)
    # scipy fallback
    if ndi is not None:
        str_el = np.ones((3,3), dtype=bool)
        for _ in range(int(erode)):
            m = ndi.binary_erosion(m.astype(bool), structure=str_el)
        for _ in range(int(open_it)):
            m = ndi.binary_opening(m.astype(bool), structure=str_el)
        for _ in range(int(close_it)):
            m = ndi.binary_closing(m.astype(bool), structure=str_el)
        return m.astype(bool)
    # no morph libs → return as-is
    return m.astype(bool)

def _remove_small(mask, min_area=0):
    if min_area <= 1:
        return mask.astype(bool)
    m = mask.astype(np.uint8)
    if cv2 is not None:
        num, lab = cv2.connectedComponents(m, connectivity=8)
        out = np.zeros_like(m)
        for i in range(1, num):
            comp = (lab == i)
            if comp.sum() >= min_area:
                out[comp] = 1
        return out.astype(bool)
    if ndi is not None:
        lab, num = ndi.label(m)
        out = np.zeros_like(m)
        for i in range(1, num+1):
            comp = (lab == i)
            if comp.sum() >= min_area:
                out[comp] = 1
        return out.astype(bool)
    # fall back: cheap area filter by counting boxes (very rough)
    return mask.astype(bool)

def refine_aux_prior(
    teacher_logits_or_probs,
    student_union_bool,
    *,
    logits_are_probs=False,
    near_thr=0.35,
    far_thr=0.65,
    max_near_px=25,
    far_cut_px=80,
    keep_only_teacher_minus_student=False,
    min_comp_area=80,
    light_erosion_iters=1,
    open_iters=0,
    close_iters=1,
    auto_tighten=True
):
    """
    Build a refined teacher prior guided by the student union:
      - lower threshold near student to recover details
      - higher threshold far from student to kill spill
      - optional teacher-only (exclude student area)
      - morphology cleanup + small component removal
      - auto-tighten if coverage explodes
    Returns:
      prior_bool, teacher_prob  (teacher_prob always in [0,1])
    """
    tp = (teacher_logits_or_probs.copy()
          if isinstance(teacher_logits_or_probs, np.ndarray) else np.array(teacher_logits_or_probs))
    if not logits_are_probs:
        tp = _sigmoid(tp)
    tp = np.clip(tp, 0.0, 1.0).astype(np.float32)

    stu = _to_bool(student_union_bool)
    H, W = tp.shape
    if stu.shape != (H, W):
        stu = _resize_bool(stu, (H, W))

    # distance: 0 on student, grows away; huge if no student
    dist = _distance_to_student(stu)

    # piecewise threshold: near (<= max_near_px), mid, far (>= far_cut_px)
    near_mask = (dist <= float(max_near_px))
    far_mask  = (dist >= float(far_cut_px))
    mid_mask  = ~(near_mask | far_mask)

    # interpolate threshold in the mid band
    thr_map = np.full_like(tp, far_thr, dtype=np.float32)
    thr_map[near_mask] = near_thr
    if mid_mask.any():
        # linear ramp from near_thr to far_thr across [max_near_px, far_cut_px]
        span = max(1.0, float(far_cut_px) - float(max_near_px))
        ramp = (dist[mid_mask] - float(max_near_px)) / span
        thr_map[mid_mask] = near_thr + ramp * (far_thr - near_thr)

    keep = (tp >= thr_map)

    if keep_only_teacher_minus_student:
        keep &= ~stu

    # morphology & area cleanup (erosion→open→close typical order)
    keep = _morphology(keep, erode=light_erosion_iters, open_it=open_iters, close_it=close_iters)
    keep = _remove_small(keep, min_comp_area)

    # emergency auto-tighten (prevents "all blue" wash)
    cov = float(keep.mean())
    if auto_tighten and cov > 0.65:
        # push thresholds toward stricter side and re-apply
        near_thr2 = min(0.80, max(near_thr, 0.50))
        far_thr2  = min(0.95, max(far_thr, 0.85))
        thr_map2 = np.full_like(tp, far_thr2, dtype=np.float32)
        thr_map2[near_mask] = near_thr2
        if mid_mask.any():
            span = max(1.0, float(far_cut_px) - float(max_near_px))
            ramp = (dist[mid_mask] - float(max_near_px)) / span
            thr_map2[mid_mask] = near_thr2 + ramp * (far_thr2 - near_thr2)
        keep2 = (tp >= thr_map2)
        if keep_only_teacher_minus_student:
            keep2 &= ~stu
        keep2 = _morphology(keep2, erode=light_erosion_iters+1, open_it=open_iters, close_it=close_iters+1)
        keep2 = _remove_small(keep2, max(min_comp_area, 150))
        # accept tightened if it actually reduces coverage
        if keep2.mean() < keep.mean():
            keep = keep2

    return keep.astype(bool), tp


def filter_student_instances_with_prior(
    pmasks_bool, scores, prior_bool,
    *, clip_to_prior=True, cover_thr=0.25, rescore_alpha=0.25
):
    """
    pmasks_bool: list of HxW bool instance masks (student)
    scores     : np.array of shape [N]
    prior_bool : HxW bool prior
    clip_to_prior: if True, we AND each mask with prior; else we drop masks with low overlap
    cover_thr  : min fraction of instance covered by prior to keep (if not clipping), or after clipping to accept
    rescore_alpha: blend factor to up/down-weight score by prior coverage
    Returns:
      preds_filt: list of dicts {"mask": bool HxW, "score": float}
      union_filt: HxW bool union of kept masks
    """
    H, W = prior_bool.shape
    preds_filt = []
    union = np.zeros((H, W), dtype=bool)

    N = len(pmasks_bool)
    if N == 0:
        return preds_filt, union

    scores = np.asarray(scores, dtype=np.float32)
    for i in range(N):
        m = pmasks_bool[i].astype(bool)
        if m.shape != (H, W):
            # resize safely
            if cv2 is not None:
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            else:
                from PIL import Image
                m = np.array(Image.fromarray(m.astype(np.uint8)*255).resize((W, H), Image.NEAREST)) > 127

        if clip_to_prior:
            m_clip = m & prior_bool
            inst_area = float(m.sum() + 1e-6)
            cover = float(m_clip.sum()) / inst_area
            if cover < cover_thr:
                continue  # drop
            new_score = float((1.0 - rescore_alpha) * scores[i] + rescore_alpha * cover)
            preds_filt.append({"mask": m_clip, "score": new_score})
            union |= m_clip
        else:
            # keep only if enough of the instance is supported by the prior
            inter = float((m & prior_bool).sum())
            inst_area = float(m.sum() + 1e-6)
            cover = inter / inst_area
            if cover < cover_thr:
                continue
            new_score = float((1.0 - rescore_alpha) * scores[i] + rescore_alpha * cover)
            preds_filt.append({"mask": m, "score": new_score})
            union |= m

    return preds_filt, union
