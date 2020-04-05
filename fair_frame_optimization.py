#!/usr/bin/env python3

from scipy import optimize
import numpy as np
import click 


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

def solve_constraint(n, rho):
    lower = 0.0
    upper = 1.0 / (1.0 / rho + n + n * rho)
    cnt = 1
    while np.abs(upper -lower) > 1e-10:
        mid = (lower + upper) / 2.0
        if constraint(n, rho, mid) < 1.0:
            lower = mid 
        else:
            upper = mid 
        cnt += 1
        if cnt > 1e3:
            break
    if constraint(n, rho, mid) < 1.0:
        return mid
    else:
        return None 

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-n', '--port-number', default=64, type=int, help='Number of (input or output) ports')
@click.option('-l', '--load-bound', default=0.9,type=float, help='Upper bound for the offerred load')
def parameterize(port_number, load_bound):
    """Calculating the best parameters for Fair-Frame algorithm"""
    result = optimize.minimize(
        lambda x: objective_func(n=port_number, rho=load_bound, delta=x),
        (1.0 / (port_number * port_number),),
        bounds=((1e-12, 1.0 / port_number),),
    )
    print(result)
    delta_min = result.x[0]
    c = constraint(port_number, load_bound, delta_min)
    print(f'delta_min: {delta_min}')
    if c < 1.0:
        print(f'constraint: {c} < 1')
    else:
        print('constraint not satisfied')
        delta_min = solve_constraint(port_number, load_bound)
        print(f'new delta_min: {delta_min}')
        c = constraint(port_number, load_bound, delta_min)
        print(f'constraint: {c} < 1')
    T = batch_size(port_number, load_bound, delta_min)
    print(f'batch size: {T}')

if __name__ == "__main__":
    parameterize()
