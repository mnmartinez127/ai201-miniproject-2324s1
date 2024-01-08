import numpy as np
from sklearn.utils.extmath import safe_sparse_dot


class CholeskySolver:

    coef_ = None
    intercept_ = np.zeros((1,))
    X_val= None
    y_val= None
    data_features = None
    validation_features = None
    def __init__(self, alpha=1e-7):
        self.alpha: float = alpha
        self.XtX = None
        self.XtY = None
        self.coef_ = None
        self.intercept_ = None

    def _reset_model(self):
        """Clean temporary data storage.
        """
        self.XtX = None
        self.XtY = None
        self.coef_ = None
        self.intercept_ = None

    def _init_model(self, X, y):
        """Initialize model storage.
        """
        d_in = X.shape[1]
        self.XtX = np.eye(d_in + 1) * self.alpha
        self.XtX[0, 0] = 0
        if len(y.shape) == 1:
            self.XtY = np.zeros((d_in + 1,))
        else:
            self.XtY = np.zeros((d_in + 1, y.shape[1]))

    def _validate_model(self, X, y):
        if self.XtX is None:
            raise RuntimeError("Model is not initialized")

        if X.shape[1] + 1 != self.XtX.shape[0]:
            n_new, n_old = X.shape[1], self.XtX.shape[0] - 1
            raise ValueError("Number of features %d does not match previous data %d." % (n_new, n_old))

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different number of samples.")

    def _update_model(self, X, y, forget=False):
        """Update model with new data; or remove already trained data from model.
        """
        X_sum = safe_sparse_dot(X.T, np.ones((X.shape[0],)))
        y_sum = safe_sparse_dot(y.T, np.ones((y.shape[0],)))

        if not forget:
            self.XtX[0, 0] += X.shape[0]
            self.XtX[1:, 0] += X_sum
            self.XtX[0, 1:] += X_sum
            self.XtX[1:, 1:] += X.T @ X

            self.XtY[0] += y_sum
            self.XtY[1:] += X.T @ y
        else:
            self.XtX[0, 0] -= X.shape[0]
            self.XtX[1:, 0] -= X_sum
            self.XtX[0, 1:] -= X_sum
            self.XtX[1:, 1:] -= X.T @ X

            self.XtY[0] -= y_sum
            self.XtY[1:] -= X.T @ y

        # invalidate previous solution
        self.coef_ = None
        self.intercept_ = None

    def compute_output_weights(self):
        """Compute solution from model with some data in it.

        Second stage of solution (X'X)B = X'Y that uses a fast Cholesky decomposition approach.
        """
        if self.XtX is None:
            raise RuntimeError("Attempting to solve uninitialized model")
        B = np.linalg.solve(self.XtX, self.XtY)
        self.coef_ = B[1:]
        self.intercept_ = B[0]

    def fit(self, X, y):
        self._reset_model()
        self._init_model(X, y)
        self._update_model(X, y)
        self.compute_output_weights()

    def partial_fit(self, X, y, compute_output_weights=True, forget=False):
        if forget:
            self.batch_forget(X, y, compute_output_weights)
        else:
            self.batch_update(X, y, compute_output_weights)

    def batch_update(self, X, y, compute_output_weights=True):
        if self.XtX is None:
            self._init_model(X, y)
        else:
            self._validate_model(X, y)

        self._update_model(X, y)
        if compute_output_weights:
            self.compute_output_weights()

    def batch_forget(self, X, y, compute_output_weights=True):
        if self.XtX is None:
            raise RuntimeError("Attempting to subtract data from uninitialized model")
        else:
            self._validate_model(X, y)

        self._update_model(X, y, forget=True)
        if compute_output_weights:
            self.compute_output_weights()
