import numpy as np
from scipy import stats


class SimpleLinearRegression:
    """
    This is a class for fitting/predicting data using ordinary least squares for simple linear regress with model and
    coefficient summaries.

    Attributes:
        intercept_ : float
            Estimated intercept of the regression fit; default is NAN prior to calling fit method.
        coefficient_ : float
            Estimated coefficient or slope of the regression fit; default is NAN prior to calling fit method.
        coefficient_metrics : dict
            Dictionary of model metrics including standard error, t-statistic, and p-value; default is an empty
            dictionary prior to calling fit method.
        model_metrics : dict
            Dictionary of coefficient metrics including residual standard error, r-squared, and adjusted r-squared;
            default is an empty dictionary prior to calling fit method.
    """

    def __init__(self):
        """
        The constructor for SimpleLinearRegression class.
        """
        self.intercept_ = np.NAN
        self.coefficient_ = np.NAN
        self.coefficient_metrics = {}
        self.model_metrics = {}

    def fit(self, X, y):
        """
        Fits the simple linear regression model using ordinary least squares and initializes both model_metrics /
        coefficient metrics.
        Parameters
        ----------
        X : 1darray
            Input feature for simple linear regression.
        y : 1darray
            Response for simple linear regression.
        Returns
        -------
            None
        """
        self.coefficient_ = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum(
            (X - np.mean(X)) ** 2
        )
        self.intercept_ = np.mean(y) - (self.coefficient_ * np.mean(X))
        self.calculate_model_metrics(X, y)
        self.calculate_coefficient_metrics(X, y)

    def predict(self, X):
        """
        Predict response using model estimates derived from the simple linear regression fit method.
        Parameters
        ----------
        X : 1darray
            Input feature for simple linear regression.
        Returns
        -------
        1darray
            Predicted response for the given input feature X.

        """
        return self.intercept_ + (self.coefficient_ * X)

    def calculate_coefficient_metrics(self, X, y):
        """
        Calculates the standard error, t-statistic, and p-value for the intercept/coefficient and stores in dictionary.
        Parameters
        ----------
        X : 1darray
            Input feature for simple linear regression.
        y : 1darray
            Response for simple linear regression.
        Returns
        -------
            None
        """
        # predictions
        y_hat = self.predict(X)
        # coefficient se
        coeff_num = (1 / (X.shape[0] - 2)) * np.sum((y_hat - y) ** 2)
        coeff_den = np.sum((X - np.mean(X)) ** 2)
        coeff_se = np.sqrt(coeff_num / coeff_den)
        # coefficient tval
        coeff_tstat = self.coefficient_ / coeff_se
        # p-value
        pval = stats.t.sf(np.abs(coeff_tstat), X.shape[0] - 1) * 2
        # add to dict
        self.coefficient_metrics["coefficient"] = {
            "coefficient": round(self.coefficient_, 2),
            "se": round(coeff_se, 2),
            "t-stat": round(coeff_tstat, 2),
            "p-val": round(pval, 2),
        }

        # intercept se
        inter_num = (1 / (X.shape[0])) + np.mean(X) ** 2
        inter_den = np.sum((X - np.mean(X)) ** 2)
        inter_se = np.sqrt((coeff_num * ((1 / (X.shape[0])) + (inter_num / inter_den))))
        # intercept tval
        inter_tstat = self.intercept_ / inter_se
        # p-value
        pval = stats.t.sf(np.abs(inter_tstat), X.shape[0] - 2) * 2
        # add to dict
        self.coefficient_metrics["intercept"] = {
            "coefficient": round(self.intercept_, 2),
            "se": round(inter_se, 2),
            "t-stat": round(inter_tstat, 2),
            "p-val": round(pval, 2),
        }

    def calculate_model_metrics(self, X, y):
        """
        Calculates the residual standard error, r-squared, and adjusted r-squared of the model and stores in dictionary.
        Parameters
        ----------
        X : 1darray
            Input feature for simple linear regression.
        y : 1darray
            Response for simple linear regression.
        Returns
        -------
            None
        """
        # RSE
        y_hat = self.predict(X)
        rss = np.sum((y - y_hat) ** 2)
        rse = np.sqrt((1 / (y.shape[0] - 2)) * rss)
        self.model_metrics["rse"] = round(rse, 2)

        # R_2
        tss = np.sum((y - np.mean(y)) ** 2)
        r_2 = (tss - rss) / tss
        self.model_metrics["r_2"] = round(r_2, 2)

        # Adjusted R_2
        adj_r_2 = 1 - (1 - r_2) * (X.shape[0] - 1) / (X.shape[0] - 1 - 1)
        self.model_metrics["adj_r_2"] = round(adj_r_2, 2)
