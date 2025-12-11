###################################

import math
import numpy as np

from numpy import meshgrid, array, unique, einsum

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy import integrate
import itertools
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.linalg import solve_sylvester
from numba import njit
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

###################################

class photon:    
    def __init__(self, x = None, y = None, sigma = 1, dim = 10, rho = None):
        self.mu = np.array([x,y])
        self.sigma = sigma # sigma_x = sigma_y by assumption

        self.dim = dim

        if rho is None:
            self.rho = np.array([
                self.psi(*self.idx2ij(idx)[0])
                for idx in range(dim**2)
            ])
            self.rho /= np.linalg.norm(self.rho)
            self.rho = np.outer(self.rho, self.rho)
        else: self.rho = rho

    def ij2idx(self, i, j): return i*(self.dim) + j
    def idx2ij(self, idx): return np.array([[idx//self.dim, idx % self.dim]])
    def get(self, i, j): return self.basis_vec[self.ij2idx(i,j)]

    def psi(self, x, y):
        # point spread function
        amplitude = (-((self.mu[0] - x)**2) - ((self.mu[1] - y)**2))
        amplitude = np.exp( amplitude  / (4*(self.sigma**2)))  / (np.sqrt(2*np.pi)*self.sigma)
        return amplitude
    
    def measure(self, num = 1, basis = None, weights = None):
        if weights is None:
            if basis is None: basis = np.identity(self.dim**2)
            # weights = np.diagonal(basis @ self.rho @ basis.conj().T).real
            weights = np.einsum('ki, ij, jk -> k', basis.conj(), self.rho, basis.T)#, optimize=True)

            # this is a fail safe...
            weights[weights < 0] = 0 # prevent numerical instability issues 
            weights = weights / weights.sum()
        else: assert basis is None

        # sampled_states = np.random.choice(np.arange(weights.shape[0]), p=weights, size = num)

        try:
            sampled_states = np.random.choice(np.arange(weights.shape[0]), p=weights, size = num)
        except:
            sampled_states = np.random.categorical(
                np.log(weights), 
                num_samples=num
            )

        return sampled_states

###################################

class photon_ensemble:
    def __init__(self, photons, brightness):
        self.photons = photons
        self.brightness = brightness / np.sum(brightness)

    def sample(self, num = 1, basis = None):
        try:

            sampled_photons = np.random.choice(np.arange(len(self.photons)), p=self.brightness, size = num)
        except:
            sampled_photons = np.random.categorical(
                np.log(self.brightness), 
                num_samples=num
            )

            # sampled_photons = np.random.choice(
            #     # np.arange(len(self.photons)), 

            #     p=self.brightness, 
            #     size = num
            # )

        return np.concatenate([
            self.photons[p_i].measure(num = cnt, basis = basis) 
            for p_i, cnt in zip(*unique(sampled_photons, return_counts = True))
        ])

###################################

def mse_paired(x_targ, y_targ, x_hat, y_hat):
    targets = np.concatenate( np.array([[x_targ], [y_targ]]), axis = 0).T
    estimates = np.concatenate( np.array([[x_hat], [y_hat]]), axis = 0).T
    cost = np.sum((targets[:, None, :] - estimates[None, :, :])**2, axis=2)
    cost = array(cost) 
    targ_indices, hat_indices = linear_sum_assignment(cost)
    optimal_costs = cost[targ_indices, hat_indices]

    return optimal_costs.sum() / len(targ_indices)

###################################

# @njit
def compute_rho(x, y, xx, yy, sigma):
    rho = np.exp((-np.square(x - xx) - np.square(y - yy) )/ (4*sigma**2)) # prop to
    rho /= np.sqrt(np.square(rho).sum())#np.linalg.norm(rho)
    rho = rho.flatten()
    rho = np.outer(rho, rho)
    return rho

###################################

def compute_weights(rho, basis, dim, num_sources, brightness):
    weights = np.zeros(dim**2)

    for p in range(num_sources):    
        weights_p = einsum(
            'ki, ij, jk -> k', 
            basis.conj(), 
            rho[p], 
            basis.T, 
            optimize=True
        ) * brightness[p]

        weights_p[weights_p < 0] = 0
        weights += weights_p

    return weights / np.sum(weights)

###################################

def compute_log_likelihood(L, weights, epsilon = 10**(-32), method = "multinomial"):
    # P(measurments | theta)

    if method == "multinomial": return np.sum(np.log(weights[L]))
    elif method == "poisson": 
        ll = 0
        for l_i, cnt in zip(*np.unique(L,return_counts=True)):
            rate = np.clip(weights[l_i], a_min = epsilon, a_max = 1)
            ll +=  cnt*np.log(rate) - rate - np.log(float(math.factorial(cnt)))
        return ll
    else: 
        assert False, "not implemented yet..."

###################################

def importance_LOTUS(
        measurements,
        x_mu_prior,
        y_mu_prior,
        x_sigma_prior,
        y_sigma_prior,
        brightness, 
        likelihood_method, 
        dim,
        basis,
        sigma,
        xx,
        yy,
        num_sources,
        importance_sampling_size,
        epsilon,
        pbar = False
):
    
    # impotance_LOTUS (The Law of the Unconscious Statistician)

    # ref_mu = np.concatenate([x_mu_prior, y_mu_prior], axis = 0)
    ref_mu = np.concatenate( np.array([x_mu_prior, y_mu_prior]), axis = 0)
    # print(x_mu_prior.shape, y_mu_prior.shape)
    ref_cov = np.identity(2*num_sources) * np.clip(np.concatenate([x_sigma_prior, y_sigma_prior], axis = 0)**2, a_min = epsilon, a_max = None)
    # ref_cov = np.identity(2*num_sources)*(sigma**2)

    # print(ref_mu.shape, ref_cov.shape)
    try: 
        samples = np.random.multivariate_normal(
            ref_mu, ref_cov, 
            size=importance_sampling_size,
        )
    except:
        samples = np.random.multivariate_normal(
            ref_mu, ref_cov, 
            shape=[importance_sampling_size],
            stream=np.cpu
        )
    # samples = multivariate_t.rvs(loc=ref_mu, shape=ref_cov, df=3, size = importance_sampling_size)

    likelihoods = []
    refs = []
    rhos = []
    theta_outer = []

    gen = samples
    if pbar: gen = tqdm(samples, desc = f"posterior importance sampling")

    for sample in gen:
        rho_h = [compute_rho(sample[p], sample[num_sources + p], xx, yy, sigma) for p in range(num_sources)]

        log_posterior = compute_log_likelihood(
            measurements, compute_weights(
                rho_h, 
                basis,
                dim, 
                num_sources, 
                brightness
            ),
            epsilon = epsilon, 
            method = likelihood_method
        )
        rhos += [rho_h[0] * brightness[0]]
        for p in range(1, num_sources): 
            rhos[-1] += rho_h[p] * brightness[p]
        theta_outer += [np.outer(sample, sample)]

        log_posterior += np.array([
            (
                norm.logpdf(sample[p], loc=x_mu_prior[p], scale=x_sigma_prior[p]),
                norm.logpdf(sample[num_sources + p], loc=y_mu_prior[p], scale=y_sigma_prior[p])
            )
            for p in range(num_sources)
        ]).sum()

        likelihoods += [log_posterior]
        refs += [np.log(multivariate_normal(mean=ref_mu, cov=ref_cov).pdf(sample))]
        # refs += [np.log(multivariate_t(loc=ref_mu, shape=ref_cov, df = 3).pdf(sample))]


    rhos = np.array(rhos)
    theta_outer = np.array(theta_outer)
    refs = np.array(refs)
    likelihoods = np.array(likelihoods)
    
    log_weights = likelihoods - refs
    norm_log_weights = log_weights - logsumexp(log_weights)
    pdf_ratio = np.exp(norm_log_weights)

    updates = {
        "x_mu_prior": np.average(samples[:, :num_sources], axis = 0, weights = pdf_ratio),
        "y_mu_prior": np.average(samples[:, num_sources:], axis = 0, weights = pdf_ratio)
    }

    updates["x_sigma_prior"] = np.sqrt(np.average(np.square(samples[:, :num_sources] - updates["x_mu_prior"]), axis = 0, weights = pdf_ratio))
    updates["y_sigma_prior"] = np.sqrt(np.average(np.square(samples[:, num_sources:] - updates["y_mu_prior"]), axis = 0, weights = pdf_ratio))
    # print(updates["x_sigma_prior"].shape, updates["y_sigma_prior"].shape)
    updates["gamma_0"] = np.average(rhos, axis = 0, weights = pdf_ratio) + np.identity(rhos.shape[1])*epsilon # EQ 6, + identity to avoid instability
    updates["est_theta_outer"] = np.average(theta_outer, axis = 0, weights = pdf_ratio)

    updates["gamma_1"] = np.average(
        samples[:, :, np.newaxis, np.newaxis] * rhos[:, np.newaxis, :, :],
        axis = 0,
        weights = pdf_ratio
    ) # EQ 6

    del samples

    return updates

###################################

def update_basis(gamma_0, gamma_1, est_theta_outer, num_sources):

    B = np.array([
        solve_sylvester(gamma_0, gamma_0, 2*gamma_1[i])
        for i in range(2*num_sources) # x,y parameters
    ]) # EQ 7

    # for i in range(2*args.num_sources):
    #     print(np.allclose(gamma_0 @ B[i] + B[i] @ gamma_0, 2*gamma_1[i]))

    G = np.zeros((2*num_sources, 2*num_sources))

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i,j] = np.trace(gamma_0 @ ((B[i] @ B[j] + B[j] @ B[i])/2)) # EQ 9

    Sigma_Q = est_theta_outer - G # EQ 8

    eigenvalues, eigenvectors = np.linalg.eig(Sigma_Q) # maybe try eigh here?
    h = eigenvectors[:, np.argmin(eigenvalues)]
    gamma_1h = np.sum([h[i] * gamma_1[i] for i in range(num_sources)], axis = 0) # EQ 23

    B_gamma = solve_sylvester(gamma_0, gamma_0, 2*gamma_1h) # Fig. 3

    _, eigenvectors = np.linalg.eigh(B_gamma) # eigen decomposition, Fig. 3
    
    assert eigenvectors.shape[0] == eigenvectors.shape[1] and eigenvectors.shape[0] == B_gamma.shape[0]

    # https://physics.stackexchange.com/questions/207234/whats-the-proof-that-sum-of-all-projection-operators-for-orthonormal-basis-give

    # ident = np.zeros(eigenvectors.shape).astype(np.complex128)

    # for v in range(eigenvectors.shape[0]):
    #     ident += np.outer(eigenvectors[:,v], eigenvectors[:, v].conj())

    # print(np.allclose(ident, np.identity(ident.shape[0])))

    return eigenvectors.T

###################################

def estimate_point_sources(
    target_x, target_y, sigma, 
    dim, num_sources, 
    shots, iterations, 
    importance_sampling_size,
    x_mu_prior_init = None, y_mu_prior_init = None,
    x_sigma_prior_init = None, y_sigma_prior_init = None,
    target_brightness = None, epsilon = 10**(-32),
    likelihood_method = "multinomial",
    basis_opt = True,
    cont_samples = False, 
    verbose = True,
):
    assert not cont_samples, "NOT IMPLEMENTED YET!"
    assert not (cont_samples and basis_opt), "Can't simulate infinite dimensions..."

    ################

    if target_brightness is None: brightness = np.ones(num_sources) / num_sources
    else: brightness = target_brightness

    bins = np.linspace(-dim/2, dim/2, dim)
    xx, yy = meshgrid(bins, bins)
    xx = np.array(xx)
    yy = np.array(yy)
    state = photon_ensemble(
        [
            photon(x = target_x[p], y = target_y[p], dim = dim, sigma = sigma, rho = compute_rho(
                target_x[p], target_y[p],
                xx, yy, sigma
            ))
            for p in range(num_sources)
        ], 
        brightness = brightness
    )

    ################

    results = {
        "iteration": [0],
        "x_mu_prior": [x_mu_prior_init], # check if None....
        "x_sigma_prior": [x_sigma_prior_init], # check if None....
        "y_mu_prior": [y_mu_prior_init], # check if None....
        "y_sigma_prior": [y_sigma_prior_init], # check if None....
    }

    results["mse"] = [mse_paired(target_x, target_y, results["x_mu_prior"][-1], results["y_mu_prior"][-1])]

    basis_ops = [np.identity(dim**2)]

    ################

    if verbose: 
        print("#"*10)
        print("Initial:")
        print("\tx:", results["x_mu_prior"][-1], "| sigma_x:", results["x_sigma_prior"][-1])
        print("\ty:", results["y_mu_prior"][-1], "| sigma_y:", results["y_sigma_prior"][-1])
        print("\tMSE:", results["mse"][-1])
        print("#"*10)

    ################

    for i in range(iterations): 
        if verbose: print(f"Iteration [{i + 1}/{iterations}]")
        measurements = state.sample(num = shots, basis = basis_ops[-1])

        updates = importance_LOTUS(
            measurements,
            results["x_mu_prior"][-1], results["y_mu_prior"][-1],
            results["x_sigma_prior"][-1], results["y_sigma_prior"][-1],
            brightness, 
            likelihood_method, 
            dim,
            basis_ops[-1],
            sigma,
            xx, yy,
            num_sources,
            importance_sampling_size,
            epsilon,
            pbar = verbose
        )

        for key in updates: 
            if key in results: results[key] += [updates[key]]

        results["iteration"] += [i + 1]
        results["mse"] += [mse_paired(target_x, target_y, results["x_mu_prior"][-1], results["y_mu_prior"][-1])]

        if basis_opt:
            basis_ops += [update_basis(
                updates["gamma_0"], 
                updates["gamma_1"], 
                updates["est_theta_outer"], 
                num_sources
            )]
        if verbose:
            print("\tx:", results["x_mu_prior"][-1], "| sigma_x:", results["x_sigma_prior"][-1])
            print("\ty:", results["y_mu_prior"][-1], "| sigma_y:", results["y_sigma_prior"][-1])
            print("\tMSE:", results["mse"][-1])
            print("#"*10)

    return results

###################################


