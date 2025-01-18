import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

from sklearn import mixture

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])


def plot_results(X, Y_, means, covariances, index, title, xmin=-9, xmax=5, ymin=-3,ymax=6):
    #splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        ax = plt.gca()
        ell.set_alpha(0.5)
        ax.add_patch(ell)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks()
    plt.yticks()
    plt.title(title)

if __name__ == "__main__":
    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    np.random.seed(0)
    mean1 = [-7,4]
    var1 = 0.5

    cat2_mean = [1,0]
    cat2_dim1_var = 0.5
    cat2_dim2_var = 1
    cat2_dim12_cov = 0.5
    cat2_cov = [[cat2_dim1_var, cat2_dim12_cov],[cat2_dim12_cov, cat2_dim2_var]]

    cat2_samples = np.random.default_rng().multivariate_normal(cat2_mean, cat2_cov, n_samples)


    #C2 = np.array([cat2dim1, cat2dim2])
    X = np.r_[
        #np.dot(np.random.randn(n_samples, 2), C2),
        cat2_samples,
        var1 * np.random.randn(n_samples, 2) + np.array(mean1)
    ]



    # Fit a Dirichlet process Gaussian mixture using a maximum of five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full").fit(X)
    plot_results(
        X,
        dpgmm.predict(X),
        dpgmm.means_,
        dpgmm.covariances_,
        0,
        "Bayesian Gaussian Mixture with a Dirichlet process prior",
    )

    plt.show()