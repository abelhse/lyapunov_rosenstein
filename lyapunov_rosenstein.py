"""
Python code to determine the largest Lyapunov exponent
using Rosenstein's algorithm: https://www.sciencedirect.com/science/article/abs/pii/016727899390009P
"""

import numpy as np
import seaborn as sns
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.spatial.distance import pdist, squareform

import tqdm

sns.set_theme()


def distance(xe: np.array, xi: np.array) -> float:
    """Euclidean distance between two position
    vectors, xe and xi.

    Parameters
    ----------
    xe : np.array
        First position vector
    xi : np.array
        Second position vector.

    Returns
    -------
    float
        Euclidean distance || xe - xi ||.
    """

    return np.sqrt(np.sum((xi - xe) ** 2))


def get_nearest_neighbour(
    i: int, X: np.ndarray, mu: float, time_steps: int, Xdist: np.ndarray
) -> int:
    """Get the nearest neighbour of xi in X excluding
    itself, and vectors whose distances are less than mu.
    The trajectories of both xi and its nearest neighbour
    is of length time_steps.

    Parameters
    ----------
    i : int
        Index of the target vector xi.
    X : np.ndarray
        Reconstructed attractor.
    mu : float
        Mean period of the orbits of X.
    time_steps : int
        Length of the trajectory.
    Xdist: np.ndarray
        Distance matrix of X.

    Returns
    -------
    int
        Index of the nearest neighbour of xi in X.
    """

    xes = np.arange(len(X) - time_steps)
    ds = Xdist[i,:len(xes)].ravel()
    ds = np.where(ds == 0, np.inf, ds)
    index = np.where((X == X[i]).all(axis=1))[0][0]
    xes_aux = np.abs(xes - index)
    ds = np.where(xes_aux < mu, np.inf, ds)

    return np.argmin(ds)


def mp_welch(ts: np.array) -> float:
    """Gets the mean periodo o ts
    using welch method.

    Parameters
    ----------
    ts : np.array
        Time-series.

    Returns
    -------
    float
        Mean period of ts.
    """

    f, Pxx = welch(ts)
    w = Pxx / np.sum(Pxx)
    mean_frequency = np.average(f, weights=w)
    return 1 / mean_frequency


def mp_periodogram(ts: np.array) -> float:
    """Gets the mean periodo o ts
    using periodogram method.

    Parameters
    ----------
    ts : np.array
        Time-series.

    Returns
    -------
    float
        Mean period of ts.
    """

    f, Pxx = periodogram(ts)
    w = Pxx / np.sum(Pxx)
    mean_frequency = np.average(f, weights=w)
    return 1 / mean_frequency


def expected_log_distance(i: int, X: np.ndarray, j: list) -> float:
    """Calculates the average of log distance at time i.

    Parameters
    ----------
    i : int
        Index representing the i-th time.
    X : np.ndarray
        Reconstructed attractor.
    j : list
        List of nearest neighbours.

    Returns
    -------
    float
        Average of the log distance at time i.
    """

    d_ji = np.array([distance(X[j[k] + i], X[k + i]) for k in range(len(X) - i)])
    return np.mean(np.log(d_ji))


def main(
    ts_path: str,
    lag: int,
    emb_dim: int,
    t_0: int,
    t_f: int,
    delta_t: float,
    method: str,
    show: bool,
) -> float:
    """Estimate the largest Lyapunov exponent
    using Rosenstein's algorithm.

    Parameters
    ----------
    ts_path : str
        Path to the time series (.npy)
    lag : int
        Lag.
    emb_dim : int
        Embedding dimension.
    t_0 : int
        Initial time range to fit the line.
    t_f : int
        Final time range to fit the line.
    delta_t : float
        Delta t used to generate the data
    method : str
        Method to find the mean period ("welch" or "periodogram")
    show : bool
        Show <ln(divergence)> plot with fitting line.

    Returns
    -------
    float
        Largest Lyapunov exponent of the system.
    """

    # Time-series
    x_obs = np.load(ts_path)

    # Mean period
    if method == "welch":
        mu = mp_welch(x_obs)
    elif method == "periodogram":
        mu = mp_periodogram(x_obs)
    else:
        raise NotImplementedError

    # ESTIMATE LYAPUNOV EXPONENT USING ROSENSTEIN APPROACH
    J = lag
    m = emb_dim

    N = len(x_obs)
    M = N - (m - 1) * J
    X = np.empty((M, m))

    # Reconstruct the atractor from one time-series
    for i in range(M):
        idx = np.arange(i, i + (m - 1) * J + 1, J)
        X[i:] = x_obs[idx]

    t_init = t_0
    t_end = t_f

    Xdist = squareform(pdist(X))

    j = []
    for i in tqdm.tqdm(range(len(X))):
        j.append(get_nearest_neighbour(i, X, mu, t_end, Xdist).item())

    mean_log_distance = np.array(
        [expected_log_distance(i, X, j) for i in range(t_init, t_end)]
    )

    deltaT = delta_t

    time = np.arange(t_init, t_end) * deltaT

    # Least-squares method
    A = np.vstack([time, np.ones(len(time))]).T
    m, c = np.linalg.lstsq(A, mean_log_distance, rcond=None)[0]

    if show:
        _ = plt.plot(
            time, mean_log_distance, label="Average divergence", color="#800080"
        )
        plt.xlabel("Time")
        plt.ylabel("< ln (divergence) >")
        _ = plt.plot(time, m * time + c, "--", label="Fitted line", color="#ff1493")
        _ = plt.legend()
        plt.show()

    return m


if __name__ == "__main__":
    lle = main(
        ts_path="lorenz.npy",
        lag=11,
        emb_dim=9,
        t_0=80,
        t_f=150,
        delta_t=0.01,
        method="welch",
        show=True,
    )

    print(f"Largest Lyapunov exponent = {lle}")
