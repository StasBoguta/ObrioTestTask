import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt


def get_poisson(lamb, num_samples):
    rng = np.random.default_rng()
    return rng.poisson(lam=lamb, size=num_samples)


def get_exp(lamb, num_samples):
    rng = np.random.default_rng()
    return rng.exponential(lamb, num_samples)


def get_normal(num_samples):
    rng = np.random.default_rng()
    return rng.normal(size=num_samples)


if __name__ == '__main__':
    poison_lam = 6
    exp_lam = 0.5
    num_samples = 200
    iterations = 2000
    poison_averages = []
    exponential_averages = []

    for i in range(iterations):
        poisson_samples = get_poisson(poison_lam, num_samples)
        exp_samples = get_exp(1/exp_lam, num_samples) # numpy exp function accept 1/lam (1/0.5 = 2) as a parameter for generating an exponential distribution with lambda=0.5
        poison_averages.append(np.mean(poisson_samples))
        exponential_averages.append(np.mean(exp_samples))

    print(f"Poison: mean {np.mean(poison_averages)}, mode {st.mode(poison_averages)}, median {np.median(poison_averages)}\n")
    print(f"Exponential: mean {np.mean(exponential_averages)}, mode {st.mode(exponential_averages)}, median {np.median(exponential_averages)}\n")
    print(f"Current Poison mean values vs theoretical values:\nstd: {np.std(poison_averages)} | {np.sqrt(poison_lam)/np.sqrt(num_samples)}\nmean: {np.mean(poison_averages)} | {poison_lam}\n")
    print(f"Current Exponential mean values vs theoretical values:\nstd: {np.std(exponential_averages)} | {1/exp_lam / np.sqrt(num_samples)}\nmean: {np.mean(exponential_averages)} | {1/exp_lam}\n")

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.title("Poison mean distribution")
    plt.hist(poison_averages, bins=100, density=True)

    plt.subplot(1, 3, 2)
    plt.hist(exponential_averages, bins=100, density=True)
    plt.title("Exponential mean distribution")

    normal_distribution = get_normal(iterations)
    plt.subplot(1, 3, 3)
    plt.hist(normal_distribution, bins=100, density=True)
    plt.title("Normal distribution")
    plt.show()
