import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn import mixture
#np.random.seed(0)

def plot_ellipse(mean,covar, color):
    v, w = linalg.eigh(covar)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
    print(angle, v, u)
    ell.set_alpha(0.3)
    return ell

def plot_cats(cat1_samples, cat2_samples):
    plt.scatter(cat1_samples[:, 0], cat1_samples[:, 1], c="gray")
    plt.scatter(cat2_samples[:, 0], cat2_samples[:, 1], c="orange")
    #plt.show()

def sample(cat_mean, cat_dim1_var, cat_dim2_var, cat_dim12_cov, n_samples):
    cat_cov = [[cat_dim1_var, cat_dim12_cov], [cat_dim12_cov, cat_dim2_var]]
    return  np.random.default_rng(0).multivariate_normal(cat_mean, cat_cov, n_samples)

def learn_and_plot(cat1_samples, cat2_samples):
    samples = np.concatenate(tuple([cat1_samples, cat2_samples]))
    dpgmm = mixture.BayesianGaussianMixture(covariance_type="full", n_components=2).fit(
        samples)
    plt.scatter(cat1_samples[:, 0], cat1_samples[:, 1], c="gray")
    plt.scatter(cat2_samples[:, 0], cat2_samples[:, 1], c="orange")
    colors = ["orange", "purple","gold","black"]
    for index,mean in enumerate(dpgmm.means_):
        cov = dpgmm.covariances_[index]
        print(index, mean, cov)
        cat_ell = plot_ellipse(mean, cov, color = colors[index])
        ax = plt.gca()
        ax.add_patch(cat_ell)
    plt.show()
    #predicted = dpgmm.predict(samples)


ba_mean = [.75,.75]
ban_mean =[1.6,.75]
ba_cov = [[0.25,0],[0,0.25]]
ban_cov = [[0.25,0],[0,0.25]]
ba_samples = np.random.default_rng(0).multivariate_normal(ba_mean, ba_cov, 70)
ban_samples = np.random.default_rng(0).multivariate_normal(ban_mean, ban_cov, 70)
plot_cats(ba_samples, ban_samples)
ell = plot_ellipse(ba_mean, ba_cov, "purple")
ax = plt.gca()
ax.add_patch(ell)
ell = plot_ellipse(ban_mean, ban_cov, "orange")
ax = plt.gca()
ax.add_patch(ell)
ell = plot_ellipse([(ban_mean[0]+ba_mean[0])/2, (ban_mean[1]+ba_mean[1])/2], [[0.75,0],[0,0.75]], "gray")
ax = plt.gca()
ax.add_patch(ell)
plt.show()
exit()
#Demonstration of overlap intuition, purely visible, no actual MOG
#Means further apart
try_mean = [.75,.75]
try_cov = [[0.25,0],[0,0.25]]
try_mean2 = [2,.75]
try_cov2 = [[0.25,0],[0,0.25]]
try_samples = np.random.default_rng(0).multivariate_normal(try_mean, try_cov, 70)
try_samples2 =  np.random.default_rng(0).multivariate_normal(try_mean2, try_cov2, 70)
plt.scatter(try_samples[:,0], try_samples[:,1], color="orange")
plt.scatter(try_samples2[:,0],try_samples2[:,1], color="gray")
plt.show()
plt.scatter(try_samples[:,0], try_samples[:,1], color="orange")
plt.scatter(try_samples2[:,0],try_samples2[:,1], color="gray")
ell = plot_ellipse(try_mean, try_cov, "orange")
ax = plt.gca()
ax.add_patch(ell)
ell = plot_ellipse(try_mean2, try_cov2, "gray")
ax = plt.gca()
ax.add_patch(ell)
plt.show()
#Means further apart
n_samples=70
try_mean = [1,.75]
try_cov = [[0.25,0],[0,0.25]]
try_mean2 = [1.5,.75]
try_cov2 = [[0.25,0],[0,0.25]]
try_samples = np.random.default_rng(0).multivariate_normal(try_mean, try_cov, 70)
try_samples2 =  np.random.default_rng(0).multivariate_normal(try_mean2, try_cov2, 70)
plt.scatter(try_samples[:,0], try_samples[:,1], color="orange")
plt.scatter(try_samples2[:,0],try_samples2[:,1], color="gray")
plt.show()
plt.scatter(try_samples[:,0], try_samples[:,1], color="orange")
plt.scatter(try_samples2[:,0],try_samples2[:,1], color="gray")
ell = plot_ellipse([(try_mean[0]+try_mean2[0])/2, (try_mean[1]+try_mean2[1])/2] ,try_cov, "gray")
ax = plt.gca()
ax.add_patch(ell)
#ell = plot_ellipse(try_mean2, try_cov2, "gray")
#ax = plt.gca()
#ax.add_patch(ell)
plt.show()
#Less overlap for low vowels
nasal_scatter_color = "orange"
nasal_label_color = '#E86100'
oral_scatter_color = "gray"
oral_label_color = "black"
bi_mean = [1, 4]
bi_dim1_var = 0.25
bi_dim2_var = .5
bi_dim12_cov = 0
bi_cov = [[bi_dim1_var, bi_dim12_cov], [bi_dim12_cov, bi_dim2_var]]
# bin
bin_mean = [1.5, 4]
bin_dim1_var = 0.25
bin_dim2_var = .5
bin_dim12_cov = 0
bin_cov = [[bin_dim1_var, bin_dim12_cov], [bin_dim12_cov, bin_dim2_var]]
# ba
ba_mean = [.5, 0.5]
ba_dim1_var = 0.25
ba_dim2_var = .5
ba_dim12_cov = 0
ba_cov = [[ba_dim1_var, ba_dim12_cov], [ba_dim12_cov, ba_dim2_var]]
# ban
ban_mean = [2, 0.5]
ban_dim1_var = 0.25
ban_dim2_var = .5
ban_dim12_cov = 0
ban_cov = [[ban_dim1_var, ban_dim12_cov], [ban_dim12_cov, ban_dim2_var]]

bi_samples = np.random.default_rng(0).multivariate_normal(bi_mean, bi_cov, n_samples)
bin_samples = np.random.default_rng(0).multivariate_normal(bin_mean, bin_cov, n_samples)
ba_samples = np.random.default_rng(0).multivariate_normal(ba_mean, ba_cov, n_samples)
ban_samples = np.random.default_rng(0).multivariate_normal(ban_mean, ban_cov, n_samples)

plt.scatter(bi_samples[:, 0], bi_samples[:, 1], c="gray")
plt.annotate("bi", (0.1, 4), fontsize=40, weight="bold")
plt.scatter(ba_samples[:, 0], ba_samples[:, 1], c="gray")
plt.annotate("ba", (0.1, 0.25), fontsize=40, weight="bold")
plt.scatter(bin_samples[:, 0], bin_samples[:, 1], c=nasal_scatter_color)
plt.annotate("bin", (2.5, 4), c=nasal_label_color, fontsize=40, weight="bold")
plt.scatter(ban_samples[:, 0], ban_samples[:, 1], c=nasal_scatter_color)
plt.annotate("ban", (2.5, 0.25), c=nasal_label_color, fontsize=40, weight="bold")

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
plt.show()

exit()

##Unlabeled data
n_samples = 70

####Less overlapping####
#Define basic distributions, mostly separated
#Distribution 1a
cat1_mean = [1, 2.5]  # [.4,.4] #
cat1_dim1_var = .15  # .6 #
cat1_dim2_var = .15  # .8 #
cat1_dim12_cov = 0  # .5 #


#Distribution 2a
cat2_mean = [4, 5.5]  # [.4,.4] #
cat2_dim1_var = .15  # .6 #
cat2_dim2_var = .15  # .8 #
cat2_dim12_cov = 0  # .5 #


#Closer
cat1_mean = [3.8, 4.8]
cat2_mean = [4, 5]

cat1_samples = sample(cat1_mean, cat1_dim1_var, cat1_dim2_var, cat1_dim12_cov, n_samples)
cat2_samples = sample(cat2_mean, cat2_dim1_var, cat2_dim2_var, cat2_dim12_cov, n_samples)
plot_cats(cat1_samples, cat2_samples)
plot_ellipse([3.9, 4.9], [[ 0.13595103, -0.02852876], [-0.02852876, 0.11461597]], color="orange")
plt.show()

#Plot distributions
cat1_samples = sample(cat1_mean, cat1_dim1_var, cat1_dim2_var, cat1_dim12_cov, n_samples)
cat2_samples = sample(cat2_mean, cat2_dim1_var, cat2_dim2_var, cat2_dim12_cov, n_samples)
plot_cats(cat1_samples, cat2_samples)
learn_and_plot(cat1_samples, cat2_samples)


#Redefine means to be closer together
cat1_mean = [1.5, 3]
cat2_mean = [3.5, 5]

cat1_samples = sample(cat1_mean, cat1_dim1_var, cat1_dim2_var, cat1_dim12_cov, n_samples)
cat2_samples = sample(cat2_mean, cat2_dim1_var, cat2_dim2_var, cat2_dim12_cov, n_samples)


plot_cats(cat1_samples, cat2_samples)
learn_and_plot(cat1_samples, cat2_samples)

#Closer
cat1_mean = [3.5, 4.5]
cat2_mean = [4, 5]

cat1_samples = sample(cat1_mean, cat1_dim1_var, cat1_dim2_var, cat1_dim12_cov, n_samples)
cat2_samples = sample(cat2_mean, cat2_dim1_var, cat2_dim2_var, cat2_dim12_cov, n_samples)
plot_cats(cat1_samples, cat2_samples)
learn_and_plot(cat1_samples, cat2_samples)





learn_and_plot(cat1_samples, cat2_samples)


cat1_mean = [3.5, 4.5]
cat2_mean = [3.5, 4.5]
cat1_samples = sample(cat1_mean, cat1_dim1_var, cat1_dim2_var, cat1_dim12_cov, n_samples)
cat2_samples = sample(cat2_mean, cat2_dim1_var, cat2_dim2_var, cat2_dim12_cov, n_samples)
plot_cats(cat1_samples, cat2_samples)
learn_and_plot(cat1_samples, cat2_samples)

learn_and_plot(cat1_samples, cat1_samples)



