import numpy as np
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' %
              (f.__name__, args, kw, (te - ts)))
        return result

    return wrap


S0 = 100  # initial stock price
K = 100  # strike price
T = 1  # time to expiration in years
r = 0.06  # annual risk-free rate
N = 3  # number of time steps
u = 1.1  # up-factor in binomial models
d = 1 / u  # ensure recombining tree
opttype = 'P'  # option type call or put


@timing
def american_slow_tree(K, T, S0, r, N, u, d, opttype='P'):
    dt = T / N
    q = (np.exp(r * T) - d) / (u - d)
    disc = np.exp(-r * dt)

    S = np.zeros(N + 1)
    for j in range(0, N + 1):
        S[j] = S0 * u ** j * d ** (N - j)

    C = np.zeros(N + 1)
    for j in range(0, N + 1):
        if opttype == 'P':
            C[j] = max(0, K - S[j])
        else:
            C[j] = max(0, S[j] - K)

    for i in np.arange(N - 1, -1, -1):
        for j in range(0, i + 1):
            S = S0 * u ** j * d ** (i - j)
            C[j] = disc * (q * C[j + 1] + (1 - q) * C[j])
            if opttype == 'P':
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j], S - K)
    return C[0]


@timing
def american_fast_tree(K, T, S0, r, N, u, d, opttype='P'):
    # precompute values
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # initialise stock prices at maturity
    S = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))

    # option payoff
    if opttype == 'P':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)

    # backward recursion through the tree
    for i in np.arange(N - 1, -1, -1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        if opttype == 'P':
            C = np.maximum(C, K - S)
        else:
            C = np.maximum(C, S - K)

    return C[0]


for N in [3, 50, 100, 1000, 5000]:
    print(american_fast_tree(K, T, S0, r, N, u, d, opttype='P'))
    print(american_slow_tree(K, T, S0, r, N, u, d, opttype='P'))
