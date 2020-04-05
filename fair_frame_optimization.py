#!/usr/bin/env python3

from scipy import optimize
import numpy as np

def batch_size(n, rho, delta):
    gamma = rho * np.exp(1.0 - rho)
    return np.ceil(np.log(2.0 * n / delta) / np.log(1.0 / gamma))

def objective_func(n, rho, delta):
    T = batch_size(n, rho, delta)
    # return 2 * T + 2 * delta * T * n * (2 + 2 * rho * T)
    return 2.0 * T + 2.0 * delta * T * n * (1.0 + (1.0 + 3.0 * rho * T + rho * rho * T * T) / (1.0 + rho * T))


def constraint(n, rho, delta):
    T = batch_size(n, rho, delta)
    return delta * (1.0 / rho + n + n * rho * T)


if __name__ == "__main__":
    n = 64.0
    rho = 0.9
    result = optimize.minimize(
        lambda x: objective_func(n=n, rho=rho, delta=x),
        (1.0 / (n * n),),
        bounds=((1e-12, 1.0 / n),),
    )
    print(result)
    print(constraint(n, rho, result.x[0]))
    print(batch_size(n, rho, result.x[0]))
