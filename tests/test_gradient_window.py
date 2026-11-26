# Copyright (c) 2026 Geoffrey Hutchison
# SPDX-License-Identifier: MIT
"""The N_max rule that sizes the gradient window: W = clamp((N_max - n)/d, W_min, n).

Capping the augmented matrix N = n + W*d, rather than the point count W, is the whole
point: a fixed W leaves N growing linearly in d (W=250 is 0.9 GB at d=8 but 7.7 GB at
d=16), while measured per-step acquisition cost is ~linear in N. These tests pin the
rule's boundaries -- especially the two escapes (explicit window, uncapped) and the
W_min floor, which is what binds once the step budget exceeds the cap.
"""

import pytest

from bouquet.config import GRADIENT_MATRIX_CAP, GRADIENT_WINDOW_MIN, RunOptions
from bouquet.surrogate import resolve_gradient_window as W


class TestResolveGradientWindow:
    def test_explicit_window_overrides_the_cap(self):
        """A positive --gradient-window pins W; the cap must not touch it.

        Benchmark arms depend on this: window_bench/acq_sweep pass fixed windows and
        would silently measure the cap's choice instead.
        """
        assert W(n_points=2000, num_dims=12, window=150, matrix_cap=4000,
                 window_min=50) == 150
        # even when the pinned window blows past the cap
        assert W(n_points=2000, num_dims=12, window=5000, matrix_cap=4000,
                 window_min=50) == 5000

    def test_cap_zero_keeps_every_gradient(self):
        """matrix_cap=0 is the pre-cap keep-all behaviour, still reachable."""
        assert W(n_points=5000, num_dims=16, window=0, matrix_cap=0,
                 window_min=50) == 0

    def test_returns_zero_while_the_cap_is_not_binding(self):
        """Low d / early n must be untouched -- 0 means "keep all" downstream.

        With n=100, d=4 the budget is (4000-100)/4 = 975 >= n, so every point's
        gradient already fits and windowing would be a pointless no-op.
        """
        assert W(n_points=100, num_dims=4, window=0, matrix_cap=4000,
                 window_min=50) == 0

    def test_binding_case_matches_the_documented_arithmetic(self):
        """The two validated points, straight from the rule."""
        assert W(n_points=2138, num_dims=8, window=0, matrix_cap=4000,
                 window_min=50) == (4000 - 2138) // 8 == 232
        assert W(n_points=2000, num_dims=12, window=0, matrix_cap=4000,
                 window_min=50) == (4000 - 2000) // 12 == 166

    def test_floor_takes_over_once_n_exceeds_the_cap(self):
        """n >= matrix_cap makes the budget negative; W_min must catch it.

        This is not an error state -- it is any run whose step budget exceeds the cap
        (C(12) = 5215 > 4000), where no window can honour it.
        """
        assert W(n_points=5215, num_dims=12, window=0, matrix_cap=4000,
                 window_min=50) == 50
        assert W(n_points=10**6, num_dims=20, window=0, matrix_cap=4000,
                 window_min=50) == 50

    def test_window_never_goes_below_the_floor(self):
        """Between "binding" and "negative" the floor still clamps from below."""
        # (4000 - 3900) / 12 = 8, under the floor of 50
        assert W(n_points=3900, num_dims=12, window=0, matrix_cap=4000,
                 window_min=50) == 50

    def test_zero_dims_is_not_a_division_error(self):
        """d=0 cannot happen from setup, but the rule must not raise if it does."""
        assert W(n_points=100, num_dims=0, window=0, matrix_cap=4000,
                 window_min=50) == 0

    def test_matrix_stays_bounded_across_d(self):
        """The property the rule exists for: N = n + W*d must not grow with d.

        A fixed window fails this -- that is why the cap replaced it.
        """
        n = 2000
        sizes = []
        for d in (5, 8, 10, 12, 16, 20):
            w = W(n_points=n, num_dims=d, window=0,
                  matrix_cap=GRADIENT_MATRIX_CAP, window_min=GRADIENT_WINDOW_MIN)
            sizes.append(n + (w if w else n) * d)
        # every size is within the cap, or explained by the floor
        floor_sizes = [n + GRADIENT_WINDOW_MIN * d for d in (5, 8, 10, 12, 16, 20)]
        for got, floor in zip(sizes, floor_sizes):
            assert got <= max(GRADIENT_MATRIX_CAP, floor)


class TestRunOptionsValidation:
    def test_defaults_carry_the_cap(self):
        opts = RunOptions()
        assert opts.gradient_matrix_cap == GRADIENT_MATRIX_CAP
        assert opts.gradient_window_min == GRADIENT_WINDOW_MIN
        assert opts.gradient_window == 0  # 0 = derive from the cap

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gradient_window": -1},
            {"gradient_matrix_cap": -1},
            {"gradient_window_min": 0},
            {"gradient_window_min": -5},
        ],
    )
    def test_out_of_range_values_are_rejected(self, kwargs):
        """Silent coercion here would change the surrogate without warning."""
        with pytest.raises(ValueError):
            RunOptions(**kwargs)
