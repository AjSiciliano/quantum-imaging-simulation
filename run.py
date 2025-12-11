###################################

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-dim","--dim", action="store", type=int, default=5, 
    help = "discrete grid dimension (dim x dim)"
)

parser.add_argument(
    "-p","--num_sources", action="store", type=int, default=3, 
    help = "number of point sources to simulate"
)

parser.add_argument(
    "-sfactor","--sfactor", action="store", type=float, default=2, 
    help = "factor of dim for sigma"
)

parser.add_argument(
    "-shots","--shots", action="store", type=int, default=1024, 
    help = "number of photons per iteration"
)

parser.add_argument(
    "-likelihood","--likelihood", action="store", type=str, default="multinomial", 
    help = "either poisson or multinomial for the likelihood"
)

parser.add_argument(
    "-iter","--iterations", action="store", type=int, default=50, 
    help = "number of iterations"
)

parser.add_argument(
    "-is_size","--importance_sampling_size", action="store", type=int, default=1024, 
    help = "number of importance sampling samples (M) to estimate functions over posterior"
)

parser.add_argument(
    "-t","--trials", action="store", type=int, default=100, 
    help = "number of example trials"
)

args = parser.parse_args()

###################################

import numpy as np

# import mlx.core as np
# np.random.random = np.random.uniform

from util import *
import pandas as pd
from scipy.spatial import distance
from pprint import pprint
import seaborn as sns

###################################

sigma = args.dim/args.sfactor
d = (args.dim/2)*0.1 
delta_d_0 = d * 0.1

###################################

full_results = {
    "source": [],
    "mse": [],
}

for trial in range(args.trials):

    print("#"*30)
    print("TRIAL:",trial)

    target_x = []
    target_y = []

    while True: 

        try: 
            coord = args.dim*(1 - 2*np.random.random(size=[2]))
        except:
            coord = args.dim*(1 - 2*np.random.random(shape=[2]))

        if np.linalg.norm(coord) < (0.375 * (args.dim/2)):
            target_x += [coord[0]]
            target_y += [coord[1]]
            break

    for _ in range(1, args.num_sources):
        while True:
            phi = 2*np.pi*np.random.random()
            delta_d = (1-2*np.random.random())*(delta_d_0/2)
            x = target_x[-1] + (d + delta_d)*np.cos(phi)
            y = target_y[-1] + (d + delta_d)*np.sin(phi)

            if np.linalg.norm(np.array([x,y])) < (0.375 * (args.dim/2)):
                c_t = np.array([[x,y]])
                c_c = np.concatenate(np.array([[target_x], [target_y]]), axis = 0).T
                if distance.cdist(c_t, c_c, 'euclidean').min() < (d - delta_d/2):
                    # print('skip', np.min(distance.cdist(c_t, c_c, 'euclidean')), (d - delta_d/2))
                    continue
                target_x += [x]
                target_y += [y]
                break

    target_x = np.array(target_x)
    target_y = np.array(target_y)

    brightness = np.ones(args.num_sources) / args.num_sources

    try:
        x_mu_prior_init = (args.dim/2) * (1 - 2*np.random.random(size=[args.num_sources]))/2
        y_mu_prior_init = (args.dim/2) * (1 - 2*np.random.random(size=[args.num_sources]))/2
    except:
        x_mu_prior_init = (args.dim/2) * (1 - 2*np.random.random(shape=[args.num_sources]))/2
        y_mu_prior_init = (args.dim/2) * (1 - 2*np.random.random(shape=[args.num_sources]))/2

    x_sigma_prior_init = np.array([args.dim/2] * args.num_sources)
    y_sigma_prior_init = np.array([args.dim/2] * args.num_sources)

    results = estimate_point_sources(
        target_x, target_y, sigma, 
        args.dim, args.num_sources, 
        args.shots, args.iterations, 
        args.importance_sampling_size,
        x_mu_prior_init = x_mu_prior_init, y_mu_prior_init = y_mu_prior_init,
        x_sigma_prior_init = x_sigma_prior_init, y_sigma_prior_init = y_sigma_prior_init,
        target_brightness = brightness, epsilon = 10**(-32),
        likelihood_method = "multinomial",
        basis_opt = True,
        cont_samples = False, 
        verbose = True,
    )

    full_results["source"] += ["quantum"]
    full_results["mse"] += [results["mse"][-1]]

    results = pd.DataFrame(results)

    ###################################

    results = estimate_point_sources(
        target_x, target_y, sigma, 
        args.dim, args.num_sources, 
        args.shots, args.iterations, 
        args.importance_sampling_size,
        x_mu_prior_init = x_mu_prior_init, y_mu_prior_init = y_mu_prior_init,
        x_sigma_prior_init = x_sigma_prior_init, y_sigma_prior_init = y_sigma_prior_init,
        target_brightness = brightness, epsilon = 10**(-32),
        likelihood_method = "multinomial",
        basis_opt = False,
        cont_samples = False, 
        verbose = True,
    )

    full_results["source"] += ["classical"]
    full_results["mse"] += [results["mse"][-1]]

    # results = pd.DataFrame(results)

    ###################################

    del results

    ###################################

full_results = pd.DataFrame(full_results)

moments = full_results.groupby('source')['mse'].agg(['mean', 'std'])
print(moments)
full_results['source'] = full_results['source'].map({
    source: f"{source.title()} (" + r"$\mu$" + f": {moments.loc[source, 'mean']:.4f}, " + r"$\sigma$" + f": {moments.loc[source, 'std']:.4f})"
    for source in ["quantum", "classical"]
})

plt.figure(figsize=(8.5, 6.5))

ax = sns.histplot(
    full_results, x="mse", hue = "source", kde = False, stat="probability",
    # log_scale = (True, False)
)

ax.set_xlabel("Mean Square Error (MSE)", fontsize=18, labelpad=10)
ax.set_ylabel("Probability", fontsize=18, labelpad=10)
legend = ax.get_legend()
legend.set_title(None)

sns.move_legend(ax, "upper right", fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig('./hist.png', dpi=300, bbox_inches='tight')

full_results.to_csv('results.csv')
