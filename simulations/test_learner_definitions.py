#Python file for unit testing the classes and functions in learner_definitions

import unittest
import numpy as np
from learner_definitions import Category, Language, Learner


#Category tests
class TestCatMethods(unittest.TestCase):
    def test_category_sample(self):
        testcat = Category.build_params(mean_nasality=1, mean_height=2, s_nasality=.5, s_height=.5, c_nasality_height=0.01)

        #Test that using the same seed gives a deterministic result
        samples = testcat.sample(seed=1, num_samples=10)
        samples_repeat = testcat.sample(seed=1, num_samples=10)
        self.assertTrue(np.array_equal(samples,samples_repeat),msg="Nondeterministic sampling behavior given seed")

        #Test that using a different seed gives a different result
        samples_diffseed = testcat.sample(seed=2, num_samples=10)
        self.assertFalse(np.array_equal(samples, samples_diffseed), msg="Different random states yielding same sample")
    #TODO test shift_sample
    #TODO test threshold_shift_sample


class TestLangMethods(unittest.TestCase):
    def test_sample(self):
        cat_1 = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=.5, s_height=.5, c_nasality_height=.1)
        cat_2 = Category.build_params(mean_nasality=2, mean_height=2, s_nasality=.5, s_height=.5, c_nasality_height=.1)
        testlang = Language(vowels=[cat_1, cat_2])

        #Test same behavior given same random seed
        samples, labels = testlang.sample(num_samples_per_cat=10, seed=1, get_labels=True)
        samples_repeat, labels_repeat = testlang.sample(num_samples_per_cat=10,seed=1, get_labels=True)
        self.assertTrue(np.array_equal(samples, samples_repeat), msg="Nondeterministic sampling behavior given seed")
        self.assertTrue(np.array_equal(labels, labels_repeat), msg="Nondeterministic labels given seed")

        #Test different behavior given different random seed
        samples_diffseed, labels_diffseed = testlang.sample(num_samples_per_cat=10, seed=2, get_labels=True)
        self.assertFalse(np.array_equal(samples, samples_diffseed), msg="Different random states yielding same sample")







if __name__ == "__main__":
    cat_1 = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=.5, s_height=.5, c_nasality_height=0)
    #Seed working for sampling
    print(cat_1.sample(num_samples=10, seed=1))
    print(cat_1.sample(num_samples=10, seed=1))
    #Bias working
    print(cat_1.shift_sample(num_samples=10, seed=1, bias=5))
