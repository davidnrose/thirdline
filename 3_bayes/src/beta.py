from scipy.stats import beta
import numpy as np


# to create the beta distribution using parameters a and b
def beta_distribution(sample,
                      prior,
                      ci_level): 

    # unpack the values
    x, n = sample
    a0, b0 = prior

    # return the posterior values
    a_post, b_post = a0 + x, b0 + (n - x)

    alpha = (1 - ci_level) / 2
    p_mean = a_post / (a_post + b_post)
    p_ci = beta.ppf([alpha, 1-alpha], a_post, b_post)

    return [a_post, b_post], p_mean, p_ci

# draws n random values from the beta distribution, using the posterior parameters above, and compares to the threshold
def beta_against_threshold(posterior,
                           threshold,
                           n_sims=2_000_000):

    # unpack posterior parameters
    a_post, b_post = posterior

    # draw random values
    rng = np.random.default_rng()
    draws = rng.beta(a_post, b_post, size=n_sims)

    # check against threshold to get the probability that the true value is above the threshold 
    prob_above = np.mean(draws > threshold)

    return prob_above

