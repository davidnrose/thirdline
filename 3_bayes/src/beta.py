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



# compile the above functions into a single function
def run_beta(sample, 
             prior,
             ci_level, 
             threshold,
             n_sims):

    # get the posterior values, posterior mean from the distribution, and the specified confidence interval
    posterior, p_mean, p_ci = beta_distribution(sample, 
                                                prior, 
                                                ci_level)

    # compare to the threshold to get the probability of exceeding the threshold
    prob_above = beta_against_threshold(posterior, 
                                        threshold,
                                        n_sims=n_sims)

    result = {
        "posterior": posterior,
        "p_mean": p_mean, 
        "p_ci": p_ci,
        "prob_above": prob_above
    }

    return result

    
