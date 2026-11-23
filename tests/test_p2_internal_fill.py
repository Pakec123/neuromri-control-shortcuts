"""Internal reporting resamples must retain the original conditional bootstrap."""
import numpy as np
import pandas as pd
import pytest

from tools.p2_internal_fill import paired_draw_mean
from neuromri.aggregate import paired_bootstrap


def test_internal_endpoint_bootstrap_matches_original_single_threshold():
    test = pd.DataFrame(dict(group_id=list('hgfedcba'), y=[1, 0, 1, 0, 1, 0, 1, 0],
                             score=[.7, .1, .2, .8, .9, .4, .5, .3]))
    original = paired_bootstrap(test, test, draws=2000, seed=6101,
                                threshold_left=.5, threshold_right=.7).set_index('metric')
    actual = paired_draw_mean(test, [.7], .5)
    for metric, row in actual.items():
        np.testing.assert_allclose(list(row.values()), original.loc[metric, ['right_minus_left', 'ci_lo', 'ci_hi']])


def test_fixed_calibration_draws_are_averaged_before_resampling():
    test = pd.DataFrame(dict(group_id=list('abcd'), y=[0, 0, 1, 1], score=[.1, .4, .5, .9]))
    result = paired_draw_mean(test, [.3, .7], .5)
    assert result['sensitivity']['estimate'] == -.25
    assert result['specificity']['estimate'] == -.25
    assert result['balanced_accuracy']['estimate'] == -.25
    with pytest.raises(ValueError, match='Duplicate'):
        paired_draw_mean(pd.concat([test, test]), [.7], .5)
    with pytest.raises(ValueError, match='finite'):
        paired_draw_mean(test, [np.nan], .5)
