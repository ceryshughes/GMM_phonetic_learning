from learner_definitions import *
import matplotlib.pyplot as plt

def define_slight_overlap_language():
    """
        Sets up a simple baseline language that should be easy for a cluster learner
        :return: a Language with 4 separable categories, named "slightoverlap4" with the language.name property
        """
    cat_1 = Category.build_params(mean_nasality=2, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_2 = Category.build_params(mean_nasality=2, mean_height=4, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_3 = Category.build_params(mean_nasality=3, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_4 = Category.build_params(mean_nasality=3, mean_height=4, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    return Language(vowels=[cat_1, cat_2, cat_3, cat_4], name="slightoverlap4")

def bias_plot(language, bias_type, arrows, show_ellipse=False, savefilename = None, stretch=False, length=False):
    bias_value = 3
    means, covariances = list(zip(*[(cat.mean, cat.cov) for cat in language.vowels]))
    for (mean, covar) in zip(means, covariances):
        print("plotting")
        X = np.random.default_rng(1).multivariate_normal(mean, covar, size=100)
        # Different color for oral and nasal context
        color = "orange" if mean[0] > 2.5 else "gray"
        if ((bias_type == "all context" and mean[1] < 4) or
                (bias_type == "nasal context" and mean[1] < 4 and mean[0] > 2.5)):
            if arrows:
                counter = 0
                for point in X:
                    if counter % 10 == 0:
                        plt.arrow(x=point[0], y=point[1], dx= bias_value, dy=0, head_width=.25, head_length=0.125, alpha=0.25, color="purple")
            else:
                X[:, 0] = X[:, 0] + bias_value
        if show_ellipse:
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
            ell.set_alpha(0.25)
            ax = plt.gca()
            ax.add_patch(ell)
        if length:
            plt.annotate("ba", (-1, 5), fontsize=20, weight="bold")
            # plt.scatter(ba_samples[:,0], ba_samples[:,1], c="gray")
            plt.annotate("ba:", (-1, 1), fontsize=20, weight="bold")
            plt.annotate("ba", (6, 5), c="orange", fontsize=20, weight="bold")
            # plt.scatter(ban_samples[:,0], ban_samples[:,1], c="gold")
            plt.annotate("ba:n", (8 if stretch else 6, 1), c="orange", fontsize=20, weight="bold")
        elif show_ellipse:
            plt.annotate("bi", (-1, 5), fontsize=20, weight="bold")
            # plt.scatter(ba_samples[:,0], ba_samples[:,1], c="gray")
            plt.annotate("ba", (-1, 1), fontsize=20, weight="bold")
            plt.annotate("bin", (6, 5), c="orange", fontsize=20, weight="bold")
            # plt.scatter(ban_samples[:,0], ban_samples[:,1], c="gold")
            plt.annotate("ban", (6, 1), c="orange", fontsize=20, weight="bold")
        if stretch:
            plt.arrow(x=2, y = 4, dx = bias_value/4, dy=0, alpha=0.25, color="black")
            plt.arrow(x=2+bias_value/2,y=1, dx= bias_value/2, dy=0, alpha=0.25, color="black")

        plt.scatter(X[:,0],X[:,1], color=color)
        plt.xlim(-3,10)
    if savefilename:
        plt.savefig(savefilename)
    plt.show()


if __name__ == "__main__":
    language = define_slight_overlap_language()

    ###All context bias###
    #Parent categories
    bias_plot(language, "no bias", False, show_ellipse=False,
              savefilename="bias_demo_plots/all_p.png")
    #Bias
    bias_plot(language, bias_type="all context", arrows=True,
              savefilename="bias_demo_plots/all_b.png")
    #Child input
    bias_plot(language, bias_type = "all context", arrows=False,
              savefilename="bias_demo_plots/all_c.png")

    ##Nasal context bias###
    # Bias
    bias_plot(language, bias_type="nasal context", arrows=True,
              savefilename="bias_demo_plots/context_b.png")
    # Child input
    bias_plot(language, bias_type="nasal context", arrows=False,
              savefilename="bias_demo_plots/context_c.png")

    bias_plot(language, bias_type="nasal context", arrows=False, stretch=True,
              length=True, show_ellipse=False, savefilename="bias_demo_plots/perception_hypothesis")

    bias_plot(language, bias_type="no bias", arrows=False, stretch=False,
              length=True, show_ellipse=False, savefilename="bias_demo_plots/perception_acoustics")






