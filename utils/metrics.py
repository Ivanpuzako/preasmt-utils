import numpy as np


def evaluate(estimator, x_test, y_test, metrics, rnd=2):
    y_pred = estimator.predict(x_test)
    for metric_name, metric_fn in metrics:
        metric_value = metric_fn(y_test, y_pred)
        metric_value = round(metric_value, rnd)
        print(f"{metric_name}: {metric_value}")


def wape_score(y_true: np.ndarray, y_pred: np.ndarray):
    err = np.abs(np.array(y_true) - np.array(y_pred)).sum()
    return err / np.sum(y_true)
