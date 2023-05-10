#!/usr/bin/env python
import matplotlib.pyplot as plt

from numpy import linspace
from scipy.integrate import odeint


class Params:
    def __init__(self, N, S, E, I, R, D, alpha, beta, gamma, mu, time):
        self.N = N
        self.S0 = S
        self.E0 = E
        self.I0 = I
        self.R0 = R
        self.D0 = D
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.time = time


def plot(S, E, I, R, D, t):
    """
    Plot the SEIRD model results.

    Parameters
    ----------
    S: numpy array
        Susceptible population
    E: numpy array
        Exposed population
    I: numpy array
        Infected population
    R: numpy array
        Recovered population
    D: numpy array
        Dead population
    time: numpy array
        Time steps

    Returns
    -------
    None
    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, facecolor='#ffffff', axisbelow=True)
    ax.plot(t, S/1000, 'blue', lw=2, label='Susceptible')
    ax.plot(t, E/1000, 'yellow', lw=2, label='Exposed')
    ax.plot(t, I/1000, 'red', lw=2, label='Infected')
    ax.plot(t, R/1000, 'green', lw=2, label='Recovered')
    ax.plot(t, D/1000, 'black', lw=2, label='Dead')

    ax.set_xlabel('Time / days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(c='gray', lw=1, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()


def simulate(params):
    """
    Simulate the SEIRD model.

    Parameters
    ----------
    params: Params object
        Model parameters

    Returns
    -------
    Tuple[numpy array]
        Tuple of numpy arrays (S, E, I, R, D) representing the populations
    """
    N = params.N
    y0 = params.S0, params.E0, params.I0, params.R0, params.D0
    alpha, beta, gamma, mu = params.alpha, params.beta, params.gamma, params.mu
    time = params.time


    def deriv(y, t, N, alpha, beta, gamma, mu):
        """
        Calculate the time derivatives of the variables in a compartmental model of disease spread.

        Parameters:
            y (list or array): the values of the variables at time t, in the order [S, E, I, R, D].
            t (float): the time at which the derivatives are being calculated.
            N (int): the total population size.
            alpha (float): the rate at which exposed individuals become infected.
            beta (float): the rate of disease transmission from infected individuals to susceptible individuals.
            gamma (float): the rate of recovery for infected individuals.
            mu (float): the rate of mortality for infected individuals.

        Returns:
            tuple: the time derivatives of S, E, I, R, and D.
        """
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = (beta * S * I / N) - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * (1 - mu) * I
        dDdt = gamma * mu * I
        return dSdt, dEdt, dIdt, dRdt, dDdt


    # Integrate the SEIRD equations over the time grid
    ret = odeint(deriv, y0, time, args=(N, alpha, beta, gamma, mu))
    # S, E, I, R, D = ret.T
    return ret.T


N = 1000
E0, I0, R0, D0 = 0, 1, 0, 0
# the whole population is initially susceptible
S0 = N
# latency rate, contact rate and recovery rate
alpha, beta, gamma, mu = 0.2, 0.2, 1.0/10, 0.3
# time grid measured in days
# Return evenly spaced numbers over a specified interval.
time = linspace(0, 160, 160) 
params = Params(N, S0, E0, I0, R0, D0, alpha, beta, gamma, mu, time)
results = simulate(params)
plot(*results, time)
