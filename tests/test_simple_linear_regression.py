import pytest
import numpy as np
import statsmodels.api as sm
from simple_linear_regression.simple_linear_regression import SimpleLinearRegression


def test_init_method():
    slr = SimpleLinearRegression()
    assert np.isnan(slr.intercept_)
    assert np.isnan(slr.coefficient_)
    assert slr.coefficient_metrics == {}
    assert slr.model_metrics == {}


test_data = [
    (
        np.array([651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]),
        np.array([23, 26, 30, 34, 43, 48, 52, 57, 58]),
    ),
    (
        np.array([2, 3, 5, 13, 8, 16, 11, 1, 9]),
        np.array([15, 28, 42, 64, 50, 90, 58, 8, 54]),
    ),
    (
        np.array(
            [
                50,
                60,
                80,
                90,
                110,
                150,
                150,
                160,
                180,
                200,
                200,
                210,
                210,
                220,
                220,
                250,
                260,
                260,
                280,
            ]
        ),
        np.array(
            [
                100,
                102,
                107,
                105,
                112,
                103,
                108,
                109,
                109,
                112,
                120,
                118,
                114,
                120,
                121,
                130,
                127,
                131,
                132,
            ]
        ),
    ),
]


@pytest.mark.parametrize("X,y", test_data)
def test_fit_method(X, y):
    x1 = sm.add_constant(X.reshape((-1, 1)))
    model = sm.OLS(y, x1).fit()
    slr = SimpleLinearRegression()
    slr.fit(X, y)
    assert round(model.params[0], 2) == round(slr.intercept_, 2)
    assert round(model.params[1], 2) == round(slr.coefficient_, 2)
    assert len(slr.coefficient_metrics.keys()) > 0
    assert len(slr.model_metrics.keys()) > 0


@pytest.mark.parametrize("X,y", test_data)
def test_predict_method(X, y):
    x1 = sm.add_constant(X.reshape((-1, 1)))
    model = sm.OLS(y, x1).fit()
    model_preds = model.predict(x1)
    slr = SimpleLinearRegression()
    slr.fit(X, y)
    slr_preds = slr.predict(X)
    comparison = np.round(model_preds, decimals=3) == np.round(slr_preds, decimals=3)
    assert comparison.all()


@pytest.mark.parametrize("X,y", test_data)
def test_calculate_model_metrics_method(X, y):
    x1 = sm.add_constant(X.reshape((-1, 1)))
    model = sm.OLS(y, x1).fit()
    slr = SimpleLinearRegression()
    slr.fit(X, y)
    assert round(model.rsquared, 2) == slr.model_metrics["r_2"]
    assert round(model.rsquared_adj, 2) == slr.model_metrics["adj_r_2"]
    assert round(np.sqrt(model.scale), 2) == slr.model_metrics["rse"]


@pytest.mark.parametrize("X,y", test_data)
def test_calculate_coefficient_metrics_method(X, y):
    x1 = sm.add_constant(X.reshape((-1, 1)))
    model = sm.OLS(y, x1).fit()
    slr = SimpleLinearRegression()
    slr.fit(X, y)
    assert round(model.params[0], 2) == slr.coefficient_metrics["intercept"]["estimate"]
    assert round(model.bse[0], 2) == slr.coefficient_metrics["intercept"]["se"]
    assert round(model.tvalues[0], 2) == slr.coefficient_metrics["intercept"]["t-stat"]
    assert round(model.pvalues[0], 2) == slr.coefficient_metrics["intercept"]["p-val"]
    assert (
        round(model.params[1], 2) == slr.coefficient_metrics["coefficient"]["estimate"]
    )
    assert round(model.bse[1], 2) == slr.coefficient_metrics["coefficient"]["se"]
    assert (
        round(model.tvalues[1], 2) == slr.coefficient_metrics["coefficient"]["t-stat"]
    )
    assert round(model.pvalues[1], 2) == slr.coefficient_metrics["coefficient"]["p-val"]
