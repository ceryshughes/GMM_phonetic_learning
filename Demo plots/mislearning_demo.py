import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from demo_plots import plot_ellipse
np.random.seed(0)

##Unlabeled data
n_samples = 70
cat1_mean = [2,1.5]
cat1_dim1_var = 0.4
cat1_dim2_var = 3
cat1_dim12_cov = 0
cat1_cov = [[cat1_dim1_var, cat1_dim12_cov], [cat1_dim12_cov,cat1_dim2_var]]

cat1_samples = np.random.default_rng(0).multivariate_normal(cat1_mean, cat1_cov, n_samples)
print(cat1_samples)
plt.scatter(cat1_samples[:,0], cat1_samples[:,1], c="orange")

cat2_mean = [1,3]
cat2_dim1_var = 0.5
cat2_dim2_var = 2
cat2_dim12_cov = -0.8
cat2_cov = [[cat2_dim1_var, cat2_dim12_cov], [cat2_dim12_cov,cat1_dim2_var]]

cat2_samples = np.random.default_rng(0).multivariate_normal(cat2_mean, cat2_cov, n_samples)
print(cat2_samples)
plt.scatter(cat2_samples[:,0], cat2_samples[:,1], c="purple")

plt.xlim(-0.2,3.7)
ax = plt.gca()
#plt.tick_params(left = False, right = False , labelleft = False ,
#                labelbottom = False, bottom = False)
ax.set_xlabel('Vowel Nasality', fontsize=20)
ax.set_ylabel('Vowel Height', fontsize=20)


##Parent categories: means and variances

cat1_ell = plot_ellipse(cat1_mean, cat1_cov, color="orange")
ax = plt.gca()
ax.add_patch(cat1_ell)

cat2_ell = plot_ellipse(cat2_mean, cat2_cov, color="purple")
ax = plt.gca()
ax.add_patch(cat2_ell)
plt.savefig("parent_samples.png")
plt.show()

ax = plt.gca()
plt.xlim(0,3.25)
plt.ylim(-1,3)
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax.set_xlabel('Vowel Nasality', fontsize=20)
ax.set_ylabel('Vowel Height', fontsize=20)
cat1_ell = plot_ellipse(cat1_mean, cat1_cov, color="orange")
ax.add_patch(cat1_ell)

cat2_ell = plot_ellipse(cat2_mean, cat2_cov, color="purple")
ax = plt.gca()
ax.add_patch(cat2_ell)
plt.savefig("parent_cats.png")
plt.show()

##Unlabeled parent samples
plt.scatter(cat2_samples[:,0], cat2_samples[:,1], c="black")
plt.scatter(cat1_samples[:,0], cat1_samples[:,1], c="black")
ax = plt.gca()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax.set_xlabel('Vowel Nasality', fontsize=20)
ax.set_ylabel('Vowel Height', fontsize=20)
plt.savefig("parent_unlabeled_samples.png")
plt.show()

##Learned categories: means and variances
samples = np.concatenate(tuple([cat1_samples, cat2_samples]))
dpgmm = mixture.BayesianGaussianMixture(n_components=2, covariance_type="full", init_params="random",random_state=1 ).fit(
    samples)
print("Learned:", dpgmm.means_)
predicted = dpgmm.predict(samples)
colors = ["orange", "purple", "green","blue","gold"]
for index,cat in enumerate(dpgmm.means_):
    learned_mean = dpgmm.means_[index]
    learned_cov = dpgmm.covariances_[index]
    ell = plot_ellipse(learned_mean, learned_cov, color=colors[index])
    ax = plt.gca()
    ax.add_patch(ell)



#plt.plot(learned_mean1[0], learned_mean1[1], marker='*', ls='none', ms=20, color="darkorange")
#plt.plot(learned_mean2[0], learned_mean2[1], marker='*', ls='none', ms=20, color="darkviolet")
plt.scatter(samples[:, 0], samples[:, 1], c="black")
#plt.scatter(samples[predicted == 1, 0], samples[predicted == 1, 1], c="orange")
#plt.scatter(cat2_samples[:, 0], cat2_samples[:, 1], c="purple")

ax = plt.gca()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax.set_xlabel('Vowel Nasality', fontsize=20)
ax.set_ylabel('Vowel Height', fontsize=20)
plt.savefig('child_cats_example.png')
plt.show()
print(dpgmm.converged_)


###Child productions for 2 cats
samples1 = np.random.default_rng(0).multivariate_normal(dpgmm.means_[0], dpgmm.covariances_[0], n_samples)
plt.scatter(samples1[:,0], samples1[:,1],c="orange")
ell = plot_ellipse(dpgmm.means_[0],dpgmm.covariances_[0], color="orange")
ax = plt.gca()
ax.add_patch(ell)
samples2 = np.random.default_rng(0).multivariate_normal(dpgmm.means_[1], dpgmm.covariances_[1], n_samples)
plt.scatter(samples2[:,0], samples2[:,1],c="purple")
ell = plot_ellipse(dpgmm.means_[1],dpgmm.covariances_[1], color="purple")
ax = plt.gca()
ax.add_patch(ell)
ax = plt.gca()
#plt.tick_params(left = False, right = False , labelleft = False ,
#                labelbottom = False, bottom = False)
ax.set_xlabel('Vowel Nasality', fontsize=20)
ax.set_ylabel('Vowel Height', fontsize=20)
plt.savefig("child_productions.png")
plt.show()