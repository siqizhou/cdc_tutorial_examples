"""
Generate the figure for the GP-MPC tutorial-paper example (tutorial_gp_mpc.tex).

Three panels: the closed-loop velocity against an active speed limit, the state
uncertainty propagated over the prediction horizon, and the resulting trade-off
between closed-loop cost and constraint violation. Run from the repository root:

    python ex6_SysID/make_tutorial_figure_gpmpc.py

The controller is not redefined here. It is read out of 6.3_GP_MPC.ipynb so that the
figure and the notebook can never drift apart.
"""

import os
import sys

import numpy as np
import casadi as ca
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO in sys.path:
    sys.path.remove(REPO)
sys.path.insert(0, REPO)

from utils.env import Env, Dynamics
from utils.simulator import Simulator
from ex2_LQR.lqr_utils import LQRController
from ex5_MPC.mpc_utils import MPCController
from ex6_SysID.SysID_utils import GenerateData, Identifier_GP, construct_gp_casadi_expression
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ---------------------------------------------------------------- controller
_nb = nbformat.read(os.path.join(HERE, "6.3_GP_MPC.ipynb"), as_version=4)
_src = next(c.source for c in _nb.cells if c.cell_type == "code" and "class GPMPCController" in c.source)
exec(compile(_src, "<6.3_GP_MPC.ipynb:GPMPCController>", "exec"))

# ---------------------------------------------------------------- setup
CASE, K_TERRAIN, SIGMA = 3, 0.01, 0.005
GAP, N_SAMPLES, SEED = (-0.5, 0.5), 150, 49
X0, XT = np.array([-0.5, 0.0]), np.array([0.5, 0.0])
V_MAX = 0.6
STATE_LBS, STATE_UBS = np.array([-2.0, -V_MAX]), np.array([2.0, V_MAX])
U_LBS, U_UBS = -8.0, 8.0
N, FREQ, T_END = 20, 10, 8
Q, R = np.diag([5, 5]), np.array([[0.1]])
BETAS = [0.0, 1.0, 2.0, 3.0]

np.random.seed(SEED)
_dg = GenerateData(p_range=[(-2.0, GAP[0]), (GAP[1], 2.0)], num_samples=N_SAMPLES, case=CASE, param=K_TERRAIN)
_dg.set_noise(mean=0.0, std=SIGMA)
p_train, h_train = _dg.generate_data()

gp = Identifier_GP(noise_std=SIGMA)
(l_opt, sf_opt), _ = gp.optimize_hyperparameters(p_train, h_train)
gp.fit(p_train, h_train)
h_gp, var_gp, dvar_gp = construct_gp_casadi_expression(gp)
print(f"GP hyperparameters: l = {l_opt:.4f}, sigma_f = {sf_opt:.5f}")

env_real = Env(CASE, X0, XT, param=K_TERRAIN, state_lbs=STATE_LBS, state_ubs=STATE_UBS,
               input_lbs=U_LBS, input_ubs=U_UBS)
dynamics_real = Dynamics(env_real)


def closed_loop_cost(states, inputs):
    err = np.asarray(states) - XT
    return float(np.sum(np.einsum('ij,jk,ik->i', err, Q, err)) + R[0, 0] * np.sum(np.asarray(inputs)**2))


def run(env_l, beta, name):
    ctrl = GPMPCController(env_l, Dynamics(env_l), Q, R, Q, N, FREQ, beta=beta, name=name, verbose=False)
    sim = Simulator(dynamics_real, ctrl, env_real, 1/FREQ, T_END)
    sim.run_simulation()
    states, inputs = np.array(sim.state_traj), np.array(sim.input_traj)
    v = states[:, 1]
    return dict(states=states, sigma=np.array(ctrl.Sigma_x_log), cost=closed_loop_cost(states, inputs),
                peak_v=v.max(), violation=max(0.0, v.max() - V_MAX))


_p = ca.MX.sym("p")
_zero = ca.Function("zero", [_p], [ca.MX(0.0)])
env_true = Env(CASE, X0, XT, symbolic_h_cov_ext=_zero, symbolic_dh_cov_ext=_zero, param=K_TERRAIN,
               state_lbs=STATE_LBS, state_ubs=STATE_UBS, input_lbs=U_LBS, input_ubs=U_UBS)
res_true = run(env_true, 0.0, "fig_true")

res = {}
for b in BETAS:
    env_l = Env(CASE, X0, XT, symbolic_h_mean_ext=h_gp, symbolic_h_cov_ext=var_gp, symbolic_dh_cov_ext=dvar_gp,
                param=K_TERRAIN, state_lbs=STATE_LBS, state_ubs=STATE_UBS, input_lbs=U_LBS, input_ubs=U_UBS)
    res[b] = run(env_l, b, f"fig_gpmpc_b{int(b)}")

print(f"\n{'controller':<22}{'peak v':>9}{'violation':>11}{'cost':>9}")
print("-" * 51)
print(f"{'true model':<22}{res_true['peak_v']:>9.4f}{res_true['violation']:>11.4f}{res_true['cost']:>9.1f}")
for b in BETAS:
    print(f"{'GP-MPC beta = '+str(int(b)):<22}{res[b]['peak_v']:>9.4f}{res[b]['violation']:>11.4f}{res[b]['cost']:>9.1f}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
})
RAMP = ["#f0a08c", "#dd6b52", "#c0392b", "#7b241c"]

fig = plt.figure(figsize=(3.5, 3.3))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.72, wspace=0.42)

# (a) closed-loop velocity
ax = fig.add_subplot(gs[0, :])
t = np.arange(len(res_true['states'])) / FREQ
ax.axhline(V_MAX, color="k", linestyle="--", linewidth=0.9, label=r"limit $v_{\max}$")
ax.plot(t, res_true['states'][:, 1], color="0.55", linewidth=0.9, label="true model")
for b, c in zip(BETAS, RAMP):
    ax.plot(t, res[b]['states'][:, 1], color=c, linewidth=1.1, label=rf"$\beta={b:.0f}$")
ax.set_xlim(0, 3.5); ax.set_ylim(-0.15, 1.08)
ax.set_xlabel("time [s]"); ax.set_ylabel("$v$")
ax.set_title("Closed-loop velocity", pad=3)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend(ncol=3, frameon=False, handlelength=1.3, columnspacing=0.9,
          handletextpad=0.4, borderpad=0.1, loc="upper center")

# (b) uncertainty propagated over the horizon
ax = fig.add_subplot(gs[1, 0])
sig = res[2.0]['sigma']
for k, style, lab in ((0, "-", "$t=0$"), (12, "--", "$t=1.2$ s")):
    ax.plot(np.arange(N + 1), np.sqrt(sig[k][:, 1]), style, color="#c0392b", linewidth=1.1, label=lab)
ax.set_xlabel("prediction step $i$"); ax.set_ylabel(r"$\sigma^v_{i|k}$")
ax.set_title("Uncertainty growth", pad=3)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend(frameon=False, handlelength=1.3, handletextpad=0.4, borderpad=0.1)

# (c) performance versus safety
ax = fig.add_subplot(gs[1, 1])
ax.plot([res[b]['violation'] for b in BETAS], [res[b]['cost'] for b in BETAS],
        "-", color="0.6", linewidth=0.8, zorder=1)
for b, c in zip(BETAS, RAMP):
    ax.plot(res[b]['violation'], res[b]['cost'], "o", color=c, markersize=4, zorder=2)
    ax.annotate(rf"$\beta={b:.0f}$", (res[b]['violation'], res[b]['cost']),
                textcoords="offset points", xytext=(4, 3), fontsize=6, color=c)
ax.axvline(0.0, color="k", linestyle=":", linewidth=0.7)
ax.set_xlim(-0.03, 0.185)
_c = [res[b]['cost'] for b in BETAS]
ax.set_ylim(min(_c) - 3, max(_c) + 6)
ax.set_xlabel("constraint violation"); ax.set_ylabel("closed-loop cost")
ax.set_title("Performance vs. safety", pad=3)
ax.grid(True, linewidth=0.4, alpha=0.5)

out = os.path.join(HERE, "tutorial_gp_mpc")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
print(f"\nwrote {out}.pdf and {out}.png")
