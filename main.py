import numpy as np
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


def calculate_average(samples):
    return np.mean(samples)


def build_hist(samples, name, bins=400):
    plt.hist(samples, bins, density=True)
    plt.title(name)
    plt.show()


if __name__ == '__main__':
    poison_lam = 6
    exp_lam = 0.5
    num_samples = 200
    iterations = 2000
    poison_averages = []
    exponential_averages = []

    for i in range(iterations):
        poisson_samples = get_poisson(poison_lam, num_samples)
        exp_samples = get_exp(exp_lam, num_samples)
        poison_averages.append(calculate_average(poisson_samples))
        exponential_averages.append(calculate_average(exp_samples))

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
