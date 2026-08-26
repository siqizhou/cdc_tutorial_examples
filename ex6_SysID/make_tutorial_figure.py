"""
Generate the figure for the tutorial-paper example (tutorial_model_learning.tex).

Four panels, all fitted to the same data: BLR with an incorrect single basis, BLR with
the correct single basis, an untuned GP, and a GP tuned by marginal likelihood. Run from
the repository root:

    python ex6_SysID/make_tutorial_figure.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ex6_SysID.SysID_utils import GenerateData, Identifier_BLR

# ---------------------------------------------------------------- experiment setup
CASE, K_TERRAIN, OMEGA = 3, 0.01, 18      # true profile h(p) = k cos(omega p)
SIGMA, N_SAMPLES = 0.005, 150             # measurement noise std, number of samples
GAP = (-0.5, 0.5)                         # unsampled interval
PLOT_XLIM = (-1.0, 1.0)                   # plotted range; models are fitted and scored on [-2, 2]
SEED = 49


def trig_basis(freq):
    """phi(p) = [1, sin(freq p), cos(freq p)]."""
    return [lambda p: np.ones_like(p),
            lambda p, k=freq: np.sin(k * p),
            lambda p, k=freq: np.cos(k * p)]


class GP:
    """Zero-mean GP with a squared-exponential kernel."""

    def __init__(self, lengthscale, signal_std, noise_std):
        self.lengthscale, self.signal_std, self.noise_std = lengthscale, signal_std, noise_std

    def _kernel(self, pa, pb):
        d = pa.reshape(-1, 1) - pb.reshape(1, -1)
        return self.signal_std**2 * np.exp(-0.5 * (d / self.lengthscale)**2)

    def log_marginal_likelihood(self, p, h):
        p, h = np.asarray(p).ravel(), np.asarray(h).ravel()
        K = self._kernel(p, p) + (self.noise_std**2 + 1e-12) * np.eye(len(p))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, h))
        return float(-0.5 * h @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(p) * np.log(2 * np.pi))

    def tune(self, p, h):
        best = (-np.inf, (self.lengthscale, self.signal_std))
        for l in np.logspace(-2, np.log10(2.0), 50):
            for s in np.logspace(np.log10(5e-4), -1, 40):
                self.lengthscale, self.signal_std = l, s
                lml = self.log_marginal_likelihood(p, h)
                if lml > best[0]:
                    best = (lml, (l, s))
        self.lengthscale, self.signal_std = best[1]
        return best[1]

    def fit(self, p, h):
        self.p_train = np.asarray(p).reshape(-1, 1)
        self.h_train = np.asarray(h).reshape(-1, 1)
        pf = self.p_train.ravel()
        K = self._kernel(pf, pf) + (self.noise_std**2 + 1e-12) * np.eye(len(pf))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.h_train.ravel()))
        return self

    def predict(self, p):
        """Mean and predictive standard deviation, including the measurement noise."""
        Ks = self._kernel(np.asarray(p).ravel(), self.p_train.ravel())
        mean = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = self.signal_std**2 - np.sum(v**2, axis=0) + self.noise_std**2
        return mean, np.sqrt(np.maximum(var, 1e-18))


def blr_predict(model, p):
    """Mean and predictive standard deviation of a fitted Identifier_BLR.

    This is Identifier_BLR.predict(), i.e. phi^T Sigma_w phi + sigma^2: the posterior
    predictive for a new observation, which already accounts for the measurement noise.
    """
    mean, std = model.predict(np.asarray(p).reshape(-1, 1))
    return mean.ravel(), std.ravel()


# ---------------------------------------------------------------- data and models
np.random.seed(SEED)
data_gen = GenerateData(p_range=[(-2.0, GAP[0]), (GAP[1], 2.0)],
                        num_samples=N_SAMPLES, case=CASE, param=K_TERRAIN)
data_gen.set_noise(mean=0.0, std=SIGMA)
p_train, h_train = data_gen.generate_data()

p_test = np.linspace(-2.0, 2.0, 801)
h_true = K_TERRAIN * np.cos(OMEGA * p_test)

blr_wrong = Identifier_BLR(trig_basis(12), sigma2=SIGMA**2)
blr_wrong.fit(p_train, h_train)
blr_right = Identifier_BLR(trig_basis(OMEGA), sigma2=SIGMA**2)
blr_right.fit(p_train, h_train)

# 'Untuned' GP: a generic default chosen without inspecting the data, i.e. a lengthscale
# of roughly an eighth of the domain and a signal std of the order of the terrain height.
gp_untuned = GP(lengthscale=0.5, signal_std=0.01, noise_std=SIGMA).fit(p_train, h_train)
gp_tuned = GP(lengthscale=0.1, signal_std=0.01, noise_std=SIGMA)
l_opt, sf_opt = gp_tuned.tune(p_train, h_train)
gp_tuned.fit(p_train, h_train)

# Left column: the well-specified / tuned models. Right column: their failure modes.
panels = [
    (blr_predict, blr_right,  r"BLR, correct basis" + "\n" + r"$\phi=[1,\sin 18p,\cos 18p]^\top$"),
    (blr_predict, blr_wrong,  r"BLR, incorrect basis" + "\n" + r"$\phi=[1,\sin 12p,\cos 12p]^\top$"),
    (GP.predict,  gp_tuned,   "GP, tuned\n" + rf"$\ell={l_opt:.2f}$, $\sigma_f={sf_opt:.4f}$"),
    (GP.predict,  gp_untuned, "GP, untuned\n" + rf"$\ell={gp_untuned.lengthscale:.2f}$, $\sigma_f={gp_untuned.signal_std:.4f}$"),
]

# ---------------------------------------------------------------- reported statistics
in_gap = (p_test > GAP[0]) & (p_test < GAP[1])
print(f"{'panel':<26}{'region':<12}{'RMS error':>11}{'pred. std':>11}{'2-sigma cov.':>14}")
print("-" * 74)
for predict, model, title in panels:
    mean, std = predict(model, p_test)
    for name, mask in (("with data", ~in_gap), ("in the gap", in_gap)):
        err, z = h_true[mask] - mean[mask], (h_true[mask] - mean[mask]) / std[mask]
        print(f"{title.splitlines()[0]:<26}{name:<12}{np.sqrt(np.mean(err**2)):>11.5f}"
              f"{np.mean(std[mask]):>11.5f}{100*np.mean(np.abs(z) < 2):>13.1f}%")
print("-" * 74)
print(f"RMS amplitude of the true profile in the gap: {np.sqrt(np.mean(h_true[in_gap]**2)):.5f}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
})

fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0), sharex=True, sharey=True)
for ax, (predict, model, title) in zip(axes.ravel(), panels):
    mean, std = predict(model, p_test)
    ax.axvspan(*GAP, color="0.85", zorder=0, linewidth=0)
    for j in (3, 2, 1):
        ax.fill_between(p_test, mean - j*std, mean + j*std, color="lightcoral",
                        alpha=0.15, linewidth=0, zorder=2,
                        label=r"$\pm 1,2,3\,\sigma_h$" if j == 3 else None)
    ax.plot(p_train, h_train, ".", color="0.35", markersize=1.4, alpha=0.7, zorder=3, label="data")
    ax.plot(p_test, mean, "-", color="firebrick", zorder=4, label=r"$\hat h(p)$")
    ax.plot(p_test, h_true, "--", color="tab:blue", linewidth=0.9, zorder=5, label=r"$h(p)$")
    ax.set_title(title, pad=3)
    ax.set_ylim(-0.028, 0.028)
    ax.set_xlim(*PLOT_XLIM)
    ax.set_yticks([-0.02, 0, 0.02])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(length=2, pad=1.5)

for ax in axes[1]:
    ax.set_xlabel("$p$")
for ax in axes[:, 0]:
    ax.set_ylabel("$h$")
handles, labels = axes[0, 0].get_legend_handles_labels()
order = [labels.index(l) for l in (r"$h(p)$", r"$\hat h(p)$", "data", r"$\pm 1,2,3\,\sigma_h$")]
fig.legend([handles[i] for i in order], [labels[i] for i in order],
           loc="lower center", ncol=4, frameon=False, handlelength=1.4,
           columnspacing=1.2, handletextpad=0.4, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.6, rect=[0, 0.055, 1, 1])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tutorial_model_learning")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=300)
print(f"\nwrote {out}.pdf and {out}.png")
