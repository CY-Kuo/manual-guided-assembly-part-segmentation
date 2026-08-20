import numpy as np

from evaluation.evaluate import pixel_metrics


def test_pixel_metrics_perfect_overlap():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    metrics = pixel_metrics(mask, mask)
    assert abs(metrics["Dice"] - 1.0) < 1e-6
    assert abs(metrics["IoU"] - 1.0) < 1e-6
    assert abs(metrics["Precision"] - 1.0) < 1e-6
    assert abs(metrics["Recall"] - 1.0) < 1e-6


def test_pixel_metrics_no_overlap():
    pred = np.array([[1, 0], [0, 0]], dtype=bool)
    gt = np.array([[0, 0], [0, 1]], dtype=bool)
    metrics = pixel_metrics(pred, gt)
    assert metrics["Dice"] < 1e-5
    assert metrics["IoU"] < 1e-5
