import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn import mixture
np.random.seed(0)

plt.rcParams.update({'axes.labelsize':36})
nasal_scatter_color = "orange"
nasal_label_color = '#E86100'
oral_scatter_color = "gray"
oral_label_color = "black"

def plot_ellipse(mean,covar, color):
    v, w = linalg.eigh(covar)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
    ell.set_alpha(0.3)
    return ell


def snapshot_sample(bi_mean,bi_var,
                    bin_mean,bin_var,
                    ba_mean, ba_var,
                    ban_mean, ban_var,
                    plot_name, n_samples, bin_allophone=False, ban_allophone=False):

    bi_dim1_var = bi_var[0]
    bi_dim2_var = bi_var[1]
    bi_dim12_cov = 0.05
    bi_cov = [[bi_dim1_var, bi_dim12_cov], [bi_dim12_cov, bi_dim2_var]]
    # bin
    bin_dim1_var = bin_var[0]
    bin_dim2_var = bin_var[1]
    bin_dim12_cov = 0.05
    bin_cov = [[bin_dim1_var, bin_dim12_cov], [bin_dim12_cov, bin_dim2_var]]
    # ba
    ba_dim1_var = ba_var[0]
    ba_dim2_var = ba_var[1]
    ba_dim12_cov = 0.05
    ba_cov = [[ba_dim1_var, ba_dim12_cov], [ba_dim12_cov, ba_dim2_var]]
    # ban
    ban_dim1_var = ban_var[0]
    ban_dim2_var = ban_var[1]
    ban_dim12_cov = 0.05
    ban_cov = [[ban_dim1_var, ban_dim12_cov], [ban_dim12_cov, ban_dim2_var]]

    bi_samples = np.random.default_rng(0).multivariate_normal(bi_mean, bi_cov, n_samples)
    bin_samples = np.random.default_rng(0).multivariate_normal(bin_mean, bin_cov, n_samples)
    ba_samples = np.random.default_rng(0).multivariate_normal(ba_mean, ba_cov, n_samples)
    ban_samples = np.random.default_rng(0).multivariate_normal(ban_mean, ban_cov, n_samples)

    ban_label = "ban" if not ban_allophone else "bãn"
    bin_label = "bin" if not bin_allophone else r'b$\mathrm{\tilde{i}}$n'
    plt.scatter(bi_samples[:, 0], bi_samples[:, 1], c="gray")
    plt.annotate("bi", (0.1, 4), fontsize=40, weight="bold")
    plt.scatter(ba_samples[:, 0], ba_samples[:, 1], c="gray")
    plt.annotate("ba", (0.1, 0.25), fontsize=40, weight="bold")
    plt.scatter(bin_samples[:, 0], bin_samples[:, 1], c=nasal_scatter_color)
    plt.annotate(bin_label, (2.2, 4.1), c=nasal_label_color, fontsize=50 if bin_allophone else 40,weight="bold")
    plt.scatter(ban_samples[:, 0], ban_samples[:, 1], c=nasal_scatter_color)
    plt.annotate(ban_label, (2.2, 1.1), c=nasal_label_color, fontsize=50 if ban_allophone else 40, weight="bold")

    if bin_allophone:
        cat_ell = plot_ellipse(bi_mean, bi_cov, color="gray")
        ax = plt.gca()
        ax.add_patch(cat_ell)

        cat_ell = plot_ellipse(bin_mean, bin_cov, color="goldenrod")
        ax = plt.gca()
        ax.add_patch(cat_ell)
    else:
        # Combined distributions over oral and nasal
        i_samples = np.concatenate((bi_samples, bin_samples), axis=0)
        # print("samples",i_samples)
        i_mean = np.mean(i_samples, axis=0)
        # print("mean",i_mean)
        i_cov = np.cov(i_samples.T)
        # print("cov",i_cov)

        cat_ell = plot_ellipse(i_mean, i_cov, color="gray")
        ax = plt.gca()
        ax.add_patch(cat_ell)
    if ban_allophone:
        cat_ell = plot_ellipse(ba_mean, ba_cov, color="gray")
        ax = plt.gca()
        ax.add_patch(cat_ell)

        cat_ell = plot_ellipse(ban_mean, ban_cov, color="goldenrod")
        ax = plt.gca()
        ax.add_patch(cat_ell)
    else:
        a_samples = np.concatenate((ba_samples, ban_samples), axis=0)
        a_mean = np.mean(a_samples, axis=0)
        a_cov = np.cov(a_samples.T)

        cat_ell = plot_ellipse(a_mean, a_cov, color="gray")
        ax = plt.gca()
        ax.add_patch(cat_ell)


    plt.xlim(0, 3.25)
    ax = plt.gca()
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    ax.set_xlabel('Vowel Nasality')
    ax.set_ylabel('Vowel Height')
    plt.savefig(plot_name)
    plt.show()
if __name__ == "__main__":
    ### How Mixture of Gaussians works: input, output, etc

    ##Unlabeled data
    n_samples = 70

    #commented out: values for multigen demo (parent --> input --> child)
    cat1_mean = [1,2.5] #[.4,.4] #
    cat1_dim1_var = 1.1#.6 #
    cat1_dim2_var = 1.2#.8 #
    cat1_dim12_cov = -.75#.5 #
    cat1_cov = [[cat1_dim1_var, cat1_dim12_cov], [cat1_dim12_cov,cat1_dim2_var]]

    cat1_samples = np.random.default_rng(0).multivariate_normal(cat1_mean, cat1_cov, n_samples)
    print(cat1_samples)
    plt.scatter(cat1_samples[:,0], cat1_samples[:,1], c="gray")
    #Commented out: values for multigen demo
    cat2_mean = [2.3,1.3]#[3, 2.7]#
    cat2_dim1_var = 1#.5#
    cat2_dim2_var =1#.5 #
    cat2_dim12_cov = 0#-.1#
    cat2_cov = [[cat2_dim1_var, cat2_dim12_cov], [cat2_dim12_cov,cat1_dim2_var]] #TODO: fix this - get rid fo cat1 repeat (cat2dim2var is being ignored)

    cat2_samples = np.random.default_rng(0).multivariate_normal(cat2_mean, cat2_cov, n_samples)
    print(cat2_samples)
    plt.scatter(cat2_samples[:,0], cat2_samples[:,1], c="gray")

    ax = plt.gca()
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    ax.set_xlabel('Vowel Nasality')
    ax.set_ylabel('Vowel Height')
    plt.title("Parent Productions", fontsize=30)
    #plt.savefig("mog_input_demo.pdf")
    plt.savefig("parent_prods_unlabeled.pdf")
    plt.show()

    cat1_ell = plot_ellipse(cat1_mean, cat1_cov, color="purple")
    ax = plt.gca()
    ax.add_patch(cat1_ell)

    cat2_ell = plot_ellipse(cat2_mean, cat2_cov, color="orange")
    ax = plt.gca()
    ax.add_patch(cat2_ell)

    plt.scatter(cat2_samples[:, 0], cat2_samples[:, 1], c="orange")
    plt.scatter(cat1_samples[:, 0], cat1_samples[:, 1], c="purple")
    plt.title("Parent Categories", fontsize=30)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.savefig('parent_cats.pdf')
    plt.show()



    ##Learned categories: means and variances
    samples = np.concatenate(tuple([cat1_samples, cat2_samples]))
    dpgmm = mixture.BayesianGaussianMixture(n_components=2, covariance_type="full", init_params="random").fit(
        samples)
    print("Learned:",dpgmm.means_)
    learned_mean1 = dpgmm.means_[0]
    learned_cov1 = dpgmm.covariances_[0]
    learned_mean2 = dpgmm.means_[1]
    learned_cov2 = dpgmm.covariances_[1]
    predicted = dpgmm.predict(samples)
    print(predicted)
    print(dpgmm.n_iter_)
    cat1_ell = plot_ellipse(learned_mean1, learned_cov1, color="orange")
    ax = plt.gca()
    ax.add_patch(cat1_ell)

    cat2_ell = plot_ellipse(learned_mean2, learned_cov2, color="purple")
    ax = plt.gca()
    ax.add_patch(cat2_ell)

    #plt.plot(learned_mean1[0], learned_mean1[1], marker='*', ls='none', ms=20,color="orangered")
    #plt.plot(learned_mean2[0], learned_mean2[1], marker='*', ls='none', ms=20, color="darkviolet")
    col = np.where(predicted == 0, 'orange', 'purple')
    #plt.scatter(samples[predicted == 0,0], samples[predicted == 0, 1],  c="orange")
    #plt.scatter(samples[predicted == 1,0], samples[predicted == 1, 1],  c="purple")
    plt.scatter(samples[:,0], samples[:,1], c=col)
    plt.title("Child Learned Categories", fontsize=30)
    #ax.set_xlabel('Vowel Nasality')
    #ax.set_ylabel('Vowel Height')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.savefig('child_cats.pdf')
    #plt.savefig('mog_learned_demo.pdf')
    plt.show()

   # exit()

    ###Demonstration of nasal vowel split
    ##Stage 1: Nasal and oral variants
    #bi
    bi_mean = [1,4]
    bi_dim1_var = 0.25
    bi_dim2_var = .5
    bi_dim12_cov = 0
    bi_cov = [[bi_dim1_var, bi_dim12_cov], [bi_dim12_cov,bi_dim2_var]]
    #bin
    bin_mean = [1.5,4]
    bin_dim1_var = 0.25
    bin_dim2_var = .5
    bin_dim12_cov = 0
    bin_cov = [[bin_dim1_var, bin_dim12_cov], [bin_dim12_cov,bin_dim2_var]]
    #ba
    ba_mean = [1,0.5]
    ba_dim1_var = 0.25
    ba_dim2_var = .5
    ba_dim12_cov = 0
    ba_cov = [[ba_dim1_var, ba_dim12_cov], [ba_dim12_cov,ba_dim2_var]]
    #ban
    ban_mean = [1.5,0.5]
    ban_dim1_var = 0.25
    ban_dim2_var = .5
    ban_dim12_cov = 0
    ban_cov = [[ban_dim1_var, ban_dim12_cov], [ban_dim12_cov,ban_dim2_var]]

    bi_samples = np.random.default_rng(0).multivariate_normal(bi_mean, bi_cov, n_samples)
    bin_samples = np.random.default_rng(0).multivariate_normal(bin_mean, bin_cov, n_samples)
    ba_samples = np.random.default_rng(0).multivariate_normal(ba_mean, ba_cov, n_samples)
    ban_samples = np.random.default_rng(0).multivariate_normal(ban_mean, ban_cov, n_samples)

    plt.scatter(bi_samples[:,0], bi_samples[:,1], c="gray")
    plt.annotate("bi",(0.1,4), fontsize=40, weight="bold")
    plt.scatter(ba_samples[:,0], ba_samples[:,1], c="gray")
    plt.annotate("ba",(0.1, 0.25), fontsize=40, weight="bold")
    plt.scatter(bin_samples[:,0], bin_samples[:,1], c=nasal_scatter_color)
    plt.annotate("bin",(2.5,4 ), c=nasal_label_color, fontsize=40, weight="bold")
    plt.scatter(ban_samples[:,0], ban_samples[:,1], c=nasal_scatter_color)
    plt.annotate("ban",(2.5,0.25 ), c=nasal_label_color,fontsize=40, weight="bold")

    #Combined distributions over oral and nasal
    i_samples = np.concatenate((bi_samples,bin_samples), axis=0)
    #print("samples",i_samples)
    i_mean = np.mean(i_samples, axis=0)
    #print("mean",i_mean)
    i_cov = np.cov(i_samples.T)
    #print("cov",i_cov)

    cat_ell = plot_ellipse(i_mean, i_cov, color="gray")
    ax = plt.gca()
    ax.add_patch(cat_ell)

    a_samples = np.concatenate((ba_samples, ban_samples), axis=0)
    a_mean = np.mean(a_samples, axis=0)
    a_cov = np.cov(a_samples.T)

    cat_ell = plot_ellipse(a_mean, a_cov, color="gray")
    ax = plt.gca()
    ax.add_patch(cat_ell)



    plt.xlim(0,3.25)
    ax = plt.gca()
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    ax.set_xlabel('Vowel Nasality')
    ax.set_ylabel('Vowel Height')
    plt.savefig("stage1_coarticulation_example.pdf")

    #plt.show()


    # ##Stage 2: Drifting apart
    # #bi
    # bi_mean = [1,4]
    # bi_dim1_var = 0.25
    # bi_dim2_var = .5
    # bi_dim12_cov = 0
    # bi_cov = [[bi_dim1_var, bi_dim12_cov], [bi_dim12_cov,bi_dim2_var]]
    # #bin
    # bin_mean = [2,4]
    # bin_dim1_var = 0.25
    # bin_dim2_var = .5
    # bin_dim12_cov = 0
    # bin_cov = [[bin_dim1_var, bin_dim12_cov], [bin_dim12_cov,bin_dim2_var]]
    # #ba
    # ba_mean = [1,0.5]
    # ba_dim1_var = 0.25
    # ba_dim2_var = .5
    # ba_dim12_cov = 0
    # ba_cov = [[ba_dim1_var, ba_dim12_cov], [ba_dim12_cov,ba_dim2_var]]
    # #ban
    # ban_mean = [2,0.5]
    # ban_dim1_var = 0.25
    # ban_dim2_var = .5
    # ban_dim12_cov = 0
    # ban_cov = [[ban_dim1_var, ban_dim12_cov], [ban_dim12_cov,ban_dim2_var]]
    #
    # bi_samples = np.random.default_rng(0).multivariate_normal(bi_mean, bi_cov, n_samples)
    # bin_samples = np.random.default_rng(0).multivariate_normal(bin_mean, bin_cov, n_samples)
    # ba_samples = np.random.default_rng(0).multivariate_normal(ba_mean, ba_cov, n_samples)
    # ban_samples = np.random.default_rng(0).multivariate_normal(ban_mean, ban_cov, n_samples)
    #
    # plt.scatter(bi_samples[:,0], bi_samples[:,1], c="gray")
    # plt.annotate("bi",(0,5), fontsize=40, weight="bold")
    # plt.scatter(ba_samples[:,0], ba_samples[:,1], c="gray")
    # plt.annotate("ba",(0, 0.25), fontsize=40, weight="bold")
    # plt.scatter(bin_samples[:,0], bin_samples[:,1], c=nasal_scatter_color)
    # plt.annotate("bin",(2.7,5 ), c=nasal_label_color, fontsize=40, weight="bold")
    # plt.scatter(ban_samples[:,0], ban_samples[:,1], c=nasal_scatter_color)
    # plt.annotate("ban",(2.7,0.25 ), c=nasal_label_color,fontsize=40, weight="bold")
    #
    # cat_ell = plot_ellipse(bi_mean, bi_cov, color="gray")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # cat_ell = plot_ellipse(bin_mean, bin_cov, color="goldenrod")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # cat_ell = plot_ellipse(ba_mean, ba_cov, color="gray")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # cat_ell = plot_ellipse(ban_mean, ban_cov, color="goldenrod")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # plt.xlim(0,3.25)
    # ax = plt.gca()
    # plt.tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = False, bottom = False)
    # ax.set_xlabel('Vowel Nasality')
    # ax.set_ylabel('Vowel Height')
    # plt.savefig("stage2_coarticulation_example.png")
    #
    # plt.show()
    # ##Stage 3: Different categories
    # bi_mean = [1,4]
    # bi_dim1_var = 0.25
    # bi_dim2_var = .5
    # bi_dim12_cov = 0
    # bi_cov = [[bi_dim1_var, bi_dim12_cov], [bi_dim12_cov,bi_dim2_var]]
    # #bin
    # bin_mean = [3,4]
    # bin_dim1_var = 0.25
    # bin_dim2_var = .5
    # bin_dim12_cov = 0
    # bin_cov = [[bin_dim1_var, bin_dim12_cov], [bin_dim12_cov,bin_dim2_var]]
    # #ba
    # ba_mean = [1,0.5]
    # ba_dim1_var = 0.25
    # ba_dim2_var = .5
    # ba_dim12_cov = 0
    # ba_cov = [[ba_dim1_var, ba_dim12_cov], [ba_dim12_cov,ba_dim2_var]]
    # #ban
    # ban_mean = [3,0.5]
    # ban_dim1_var = 0.25
    # ban_dim2_var = .5
    # ban_dim12_cov = 0
    # ban_cov = [[ban_dim1_var, ban_dim12_cov], [ban_dim12_cov,ban_dim2_var]]
    #
    # bi_samples = np.random.default_rng(0).multivariate_normal(bi_mean, bi_cov, n_samples)
    # bin_samples = np.random.default_rng(0).multivariate_normal(bin_mean, bin_cov, n_samples)
    # ba_samples = np.random.default_rng(0).multivariate_normal(ba_mean, ba_cov, n_samples)
    # ban_samples = np.random.default_rng(0).multivariate_normal(ban_mean, ban_cov, n_samples)
    #
    # plt.scatter(bi_samples[:,0], bi_samples[:,1], c="gray")
    # plt.annotate("bi",(0,5), fontsize=40, weight="bold")
    # plt.scatter(ba_samples[:,0], ba_samples[:,1], c="gray")
    # plt.annotate("ba",(0, 0.25), fontsize=40, weight="bold")
    # plt.scatter(bin_samples[:,0], bin_samples[:,1], c=nasal_scatter_color)
    # plt.annotate("bĩn",(2.7,5 ), c=nasal_label_color, fontsize=40, weight="bold")
    # plt.scatter(ban_samples[:,0], ban_samples[:,1], c=nasal_scatter_color)
    # plt.annotate("bãn",(2.7,0.25 ), c=nasal_label_color,fontsize=40, weight="bold")
    #
    # cat_ell = plot_ellipse(bi_mean, bi_cov, color="gray")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # cat_ell = plot_ellipse(bin_mean, bin_cov, color="goldenrod")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # cat_ell = plot_ellipse(ba_mean, ba_cov, color="gray")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # cat_ell = plot_ellipse(ban_mean, ban_cov, color="goldenrod")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # plt.xlim(0,4.25)
    # ax = plt.gca()
    # plt.tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = False, bottom = False)
    # ax.set_xlabel('Vowel Nasality')
    # ax.set_ylabel('Vowel Height')
    # plt.savefig("stage3_nasal_allophones_example.pdf")
    #
    # plt.show()



    ###Adding phonetic bias


    ##High and low vowels with no bias
    # # bi
    # bi_mean = [1, 4]
    # bi_dim1_var = 0.25
    # bi_dim2_var = .5
    # bi_dim12_cov = 0
    # bi_cov = [[bi_dim1_var, bi_dim12_cov], [bi_dim12_cov, bi_dim2_var]]
    # # bin
    # bin_mean = [1.5, 4]
    # bin_dim1_var = 0.25
    # bin_dim2_var = .5
    # bin_dim12_cov = 0
    # bin_cov = [[bin_dim1_var, bin_dim12_cov], [bin_dim12_cov, bin_dim2_var]]
    # # ba
    # ba_mean = [1, 0.5]
    # ba_dim1_var = 0.25
    # ba_dim2_var = .5
    # ba_dim12_cov = 0
    # ba_cov = [[ba_dim1_var, ba_dim12_cov], [ba_dim12_cov, ba_dim2_var]]
    # # ban
    # ban_mean = [1.5, 0.5]
    # ban_dim1_var = 0.25
    # ban_dim2_var = .5
    # ban_dim12_cov = 0
    # ban_cov = [[ban_dim1_var, ban_dim12_cov], [ban_dim12_cov, ban_dim2_var]]
    #
    # bi_samples = np.random.default_rng(0).multivariate_normal(bi_mean, bi_cov, n_samples)
    # bin_samples = np.random.default_rng(0).multivariate_normal(bin_mean, bin_cov, n_samples)
    # ba_samples = np.random.default_rng(0).multivariate_normal(ba_mean, ba_cov, n_samples)
    # ban_samples = np.random.default_rng(0).multivariate_normal(ban_mean, ban_cov, n_samples)
    #
    # plt.scatter(bi_samples[:, 0], bi_samples[:, 1], c="gray")
    # #plt.annotate("bi", (0.1, 4), fontsize=40, weight="bold")
    # plt.scatter(ba_samples[:, 0], ba_samples[:, 1], c="gray")
    # #plt.annotate("ba", (0.1, 0.25), fontsize=40, weight="bold")
    # plt.scatter(bin_samples[:, 0], bin_samples[:, 1], c=nasal_scatter_color)
    # #plt.annotate("bin", (2.5, 4), c=nasal_label_color, fontsize=40, weight="bold")
    # plt.scatter(ban_samples[:, 0], ban_samples[:, 1], c=nasal_scatter_color)
    # #plt.annotate("ban", (2.5, 0.25), c=nasal_label_color, fontsize=40, weight="bold")
    #
    # # Combined distributions over oral and nasal
    # i_samples = np.concatenate((bi_samples, bin_samples), axis=0)
    # print("samples", i_samples)
    # i_mean = np.mean(i_samples, axis=0)
    # print("mean", i_mean)
    # i_cov = np.cov(i_samples.T)
    # print("cov", i_cov)
    #
    # cat_ell = plot_ellipse(i_mean, i_cov, color="gray")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # a_samples = np.concatenate((ba_samples, ban_samples), axis=0)
    # a_mean = np.mean(a_samples, axis=0)
    # a_cov = np.cov(a_samples.T)
    #
    # cat_ell = plot_ellipse(a_mean, a_cov, color="gray")
    # ax = plt.gca()
    # ax.add_patch(cat_ell)
    #
    # plt.xlim(0,4)
    # ax = plt.gca()
    # plt.tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = False, bottom = False)
    # ax.set_xlabel('Vowel Nasality')
    # ax.set_ylabel('Vowel Height')

    ##Arrows
    counter = 0
    for point in ba_samples:
        if counter % 2 == 0:
            plt.arrow(x=point[0], y=point[1], dx=0.75, dy=0, color="purple", head_width=.25, head_length=0.125, alpha=0.25)
        #counter += 1
    for point in ban_samples:
        if counter % 2 == 0:
            plt.arrow(x=point[0], y=point[1], dx=0.75, dy=0, color="purple", head_width=.25, head_length=0.125, alpha = 0.25)
        #counter += 1
    plt.savefig("arrows.pdf")
    plt.show()

    ##Biased plot
    bi_mean = [1,4]
    bi_dim1_var = 0.25
    bi_dim2_var = .5
    bi_dim12_cov = 0
    bi_cov = [[bi_dim1_var, bi_dim12_cov], [bi_dim12_cov,bi_dim2_var]]
    #bin
    bin_mean = [1.5,4]
    bin_dim1_var = 0.25
    bin_dim2_var = .5
    bin_dim12_cov = 0
    bin_cov = [[bin_dim1_var, bin_dim12_cov], [bin_dim12_cov,bin_dim2_var]]
    #ba
    ba_mean = [1,0.5]
    ba_dim1_var = 0.25
    ba_dim2_var = .5
    ba_dim12_cov = 0
    ba_cov = [[ba_dim1_var, ba_dim12_cov], [ba_dim12_cov,ba_dim2_var]]
    #ban
    ban_mean = [1.5,0.5]
    ban_dim1_var = 0.25
    ban_dim2_var = .5
    ban_dim12_cov = 0
    ban_cov = [[ban_dim1_var, ban_dim12_cov], [ban_dim12_cov,ban_dim2_var]]

    bi_samples = np.random.default_rng(0).multivariate_normal(bi_mean, bi_cov, n_samples)
    bin_samples = np.random.default_rng(0).multivariate_normal(bin_mean, bin_cov, n_samples)
    ba_samples = np.random.default_rng(0).multivariate_normal(ba_mean, ba_cov, n_samples)
    ban_samples = np.random.default_rng(0).multivariate_normal(ban_mean, ban_cov, n_samples)



    plt.scatter(bi_samples[:,0], bi_samples[:,1], c="gray")
    plt.annotate("bi",(0,4.5), fontsize=40, weight="bold")
    #plt.scatter(ba_samples[:,0], ba_samples[:,1], c="gray")
    plt.annotate("ba",(0.5, 0.25), fontsize=40,weight="bold")
    plt.scatter(bin_samples[:,0], bin_samples[:,1], c=nasal_scatter_color)
    plt.annotate("bin",(2.7,4.5 ), c=nasal_label_color, fontsize=40, weight="bold")
    #plt.scatter(ban_samples[:,0], ban_samples[:,1], c="gold")
    plt.annotate("ban",(3,0.25 ), c=nasal_label_color,fontsize=40, weight="bold")

    plt.xlim(0,4)
    ax = plt.gca()
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    ax.set_xlabel('Vowel Nasality')
    ax.set_ylabel('Vowel Height')

    ##Moved points
    shift = 0.75
    counter = 0
    for point in ba_samples:
        plt.scatter(point[0]+shift, y=point[1], color="gray")
    for point in ban_samples:
        plt.scatter(point[0]+shift, y=point[1], color=nasal_scatter_color)

    # Combined distributions over oral and nasal
    i_samples = np.concatenate((bi_samples, bin_samples), axis=0)
    #print("samples", i_samples)
    i_mean = np.mean(i_samples, axis=0)
    #print("mean", i_mean)
    i_cov = np.cov(i_samples.T)
    #print("cov", i_cov)

    cat_ell = plot_ellipse(i_mean, i_cov, color="gray")
    ax = plt.gca()
    ax.add_patch(cat_ell)

    a_samples = np.concatenate((ba_samples, ban_samples), axis=0)
    a_mean = np.mean(a_samples, axis=0)
    a_mean = [a_mean[0]+shift, a_mean[1]]
    a_cov = np.cov(a_samples.T)

    cat_ell = plot_ellipse(a_mean, a_cov, color="gray")
    ax = plt.gca()
    ax.add_patch(cat_ell)

    plt.savefig("biased_distro.pdf")
    plt.show()


    ### Sound change with contrast for just bi/bin
    snapshot_sample(bi_mean=[1,4], bi_var=[0.25, 0.25],
                    bin_mean=[1.75,4], bin_var=[0.25,0.25],
                    ba_mean=[1,0.5],ba_var=[0.25,0.25],
                    ban_mean=[1.75,0.5], ban_var=[0.25,0.25], n_samples=n_samples,
                    plot_name="starting_stage_bigger_diff.pdf")
    snapshot_sample(bi_mean=[1,4], bi_var=[0.25, 0.25],
                    bin_mean=[2.5,4], bin_var=[0.25,0.25],
                    ba_mean=[1,0.5],ba_var=[0.25,0.25],
                    ban_mean=[1.75,0.5], ban_var=[0.25,0.25], n_samples=n_samples,
                    plot_name="nasal_i_allophone.pdf", bin_allophone=True)
    snapshot_sample(bi_mean=[1,4], bi_var=[0.25, 0.25],
                    bin_mean=[1.75,4], bin_var=[0.25,0.25],
                    ba_mean=[1,0.5],ba_var=[0.25,0.25],
                    ban_mean=[2.5,0.5], ban_var=[0.25,0.25], n_samples=n_samples,
                    plot_name="nasal_a_allophone.pdf", ban_allophone=True)


