import numpy as np

from maps.smart_prior import refine_aux_prior, filter_student_instances_with_prior


def test_refine_aux_prior_returns_boolean_mask_and_probability():
    logits = np.zeros((8, 8), dtype=np.float32)
    student = np.zeros((8, 8), dtype=bool)
    student[3:5, 3:5] = True
    mask, prob = refine_aux_prior(
        logits,
        student,
        auto_tighten=False,
        min_comp_area=0,
        light_erosion_iters=0,
        close_iters=0,
    )
    assert mask.shape == (8, 8)
    assert mask.dtype == bool
    assert prob.shape == (8, 8)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_filter_student_instances_returns_union():
    prior = np.ones((6, 6), dtype=bool)
    masks = [np.eye(6, dtype=bool)]
    preds, union = filter_student_instances_with_prior(
        masks, np.array([0.9], dtype=np.float32), prior
    )
    assert len(preds) == 1
    assert union.shape == prior.shape
    assert union.any()
