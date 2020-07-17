from simple_linear_regression import SimpleLinearRegression
import numpy as np


def main():

    # example data
    X = np.array([651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518])
    y = np.array([23, 26, 30, 34, 43, 48, 52, 57, 58])

    # initialize object
    slr = SimpleLinearRegression()

    # fit simple linear regression
    slr.fit(X, y)

    # predict
    preds = slr.predict(np.array([700, 850, 1200]))
    print("Preds: ", preds)

    # print metrics
    print(slr.coefficient_metrics)
    print(slr.model_metrics)


if __name__ == "__main__":
    main()
