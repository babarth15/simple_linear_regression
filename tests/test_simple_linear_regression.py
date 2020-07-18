import numpy as np
from simple_linear_regression.simple_linear_regression import SimpleLinearRegression

# test_data = [
#     ("1", 1),
#     ("2", 2),
#     ("3", 3),
#     ("4", 4),
#     ("5", 5),
#     ("101", 101),
#     ("102", 102),
#     ("103", 103),
#     ("104", 104),
#     ("105", 105),
#     ("TOTAL_HDP_SITES", 99),
#     ("TOTAL_SP_SITES", 199),
# ]
#
#
# @pytest.mark.parametrize("org,expected", test_data)
# def test_init(org, expected):
#     assert helpers.determine_org(org) == expected


def test_init():
    slr = SimpleLinearRegression()
    assert np.isnan(slr.intercept_)
    assert np.isnan(slr.coefficient_)
    assert slr.coefficient_metrics == {}
    assert slr.model_metrics == {}
