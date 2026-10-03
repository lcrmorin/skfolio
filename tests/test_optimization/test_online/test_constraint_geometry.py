"""Constrained updates must preserve the geometry of the numerical kernel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._projection import (
    AutoProjector,
    ProjectionConfig,
    project_box_and_sum,
    project_convex,
)


def test_entropy_projection_preserves_free_ratios():
    projector = AutoProjector(ProjectionConfig(upper=0.5))
    result = projector.project_geometry(np.array([0.8, 0.15, 0.05]), "entropy")
    assert_allclose(result, [0.5, 0.375, 0.125], atol=1e-10)
    assert_allclose(
        project_convex([0.8, 0.15, 0.05], upper=0.5, geometry="entropy"),
        result,
        atol=2e-6,
    )


def test_short_exposure_is_an_upper_bound():
    result = project_convex([1.8, -0.8], lower=-1, upper=2, max_short=0.2)
    assert_allclose(result, [1.2, -0.2], atol=1e-7)


def test_variance_and_tracking_constraints_both_apply():
    result = project_convex(
        [1, 0],
        covariance=np.eye(2),
        variance_bound=0.51,
        X=np.array([[0.01, -0.01], [-0.01, 0.01]]),
        tracking_error_benchmark=np.zeros(2),
        max_tracking_error=1,
    )
    assert result @ result <= 0.51 + 1e-7


def test_infeasible_fast_budget_raises():
    with pytest.raises(ValueError, match="Infeasible"):
        project_box_and_sum([0.5, 0.5], lower=0, upper=0.2)


def test_no_budget_does_not_impose_unit_sum():
    result = AutoProjector(ProjectionConfig(budget=None)).project(np.array([0.1, 0.2]))
    assert_allclose(result, [0.1, 0.2], atol=1e-7)
