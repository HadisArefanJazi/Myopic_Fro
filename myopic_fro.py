import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import savemat


def symm(M):
    return 0.5 * (M + M.T)


def trace_fn(P):
    return np.trace(P)


def offdiag_fn(P):
    return 2.0 * abs(P[0, 1])


def fro_fn(P):
    return np.linalg.norm(P, ord="fro")


def vec2str(v, ndig):
    v = np.ravel(v)
    fmt = f"{{:.{ndig}f}}"
    return "[" + ", ".join(fmt.format(x) for x in v) + "]"


def run_myopic_rollout(P0, A_grid, T, choose_cost_fn):
    P_seq = [None] * (T + 1)
    A_seq = [None] * T
    trace_seq = np.zeros(T)
    inst_cost_seq = np.zeros(T)

    P_seq[0] = symm(P0.copy())
    I2 = np.eye(2)

    for t in range(T):
        P_prev = P_seq[t]
        best_val = np.inf
        best_A = None
        best_P = None

        for A in A_grid:
            # P_t = (P_{t-1}^{-1} + A_t)^(-1)
            P_next = np.linalg.solve(I2 + P_prev @ A, P_prev)
            P_next = symm(P_next)

            val = choose_cost_fn(P_next)
            if val < best_val:
                best_val = val
                best_A = A
                best_P = P_next

        A_seq[t] = best_A
        P_seq[t + 1] = best_P
        trace_seq[t] = np.trace(best_P)
        inst_cost_seq[t] = choose_cost_fn(best_P)

    return P_seq, A_seq, trace_seq, inst_cost_seq


def build_actions_fro_fixed_attention(tau, l1_grid, l3_grid):
    # Build A = L L' with
    #   L = [[l1, 0],
    #        [l2, l3]]
    # satisfying ||A||_F = tau
    A_grid = []
    tol = 1e-12

    for l1 in l1_grid:
        l1sq = l1 ** 2
        l1_4 = l1sq ** 2

        for l3 in l3_grid:
            l3sq = l3 ** 2
            l3_4 = l3sq ** 2

            # Let x = l2^2. Then:
            # x^2 + 2(l1^2 + l3^2)x + (l1^4 + l3^4 - tau^2) = 0
            a = 1.0
            b = 2.0 * (l1sq + l3sq)
            c = l1_4 + l3_4 - tau ** 2

            disc = b ** 2 - 4.0 * a * c
            if disc < 0:
                continue

            x = (-b + np.sqrt(disc)) / (2.0 * a)
            if x < -tol:
                continue

            l2abs = np.sqrt(max(x, 0.0))

            if l2abs <= tol or l1 <= tol:
                l2 = l2abs
                A = np.array([[l1sq, l1 * l2],
                              [l1 * l2, x + l3sq]], dtype=float)
                A_grid.append(A)
            else:
                l2 = l2abs
                A1 = np.array([[l1sq, l1 * l2],
                               [l1 * l2, x + l3sq]], dtype=float)

                l2 = -l2abs
                A2 = np.array([[l1sq, l1 * l2],
                               [l1 * l2, x + l3sq]], dtype=float)

                A_grid.append(A1)
                A_grid.append(A2)

    if not A_grid:
        raise RuntimeError(f"The admissible action set is empty for tau = {tau:.6f}.")

    return A_grid


def to_object_array_2d(x):
    arr = np.empty((len(x), len(x[0])), dtype=object)
    for i in range(len(x)):
        for j in range(len(x[0])):
            arr[i, j] = x[i][j]
    return arr


if __name__ == "__main__":
    # ================================================================
    # Myopic policy: Frobenius attention measure
    #
    # This version computes and prints for every gamma:
    #   1) Total benchmark cost for each scenario:
    #         J^[m](gamma) = sum_t Tr(P_t^[m])
    #   2) Instantaneous stage-cost sequence for each scenario:
    #         Scenario 1: Tr(P_t^[1])
    #         Scenario 2: sum_{i ~= j} |P_ij^[2]|
    #         Scenario 3: ||P_t^[3]||_F
    #   3) Stagewise regret sequence for each scenario:
    #         Regret^[m](gamma,t) =
    #            (Tr(P_t^[m]) - Tr(P_t^[1])) / Tr(P_t^[1])
    #
    # Also plots:
    #   - total benchmark cost vs gamma
    #   - stagewise regret at gamma = 0.5
    #   - instantaneous stage costs at gamma_inst
    #
    # Exact Frobenius attention constraint:
    #   G(P_t,P_{t-1}) = ||P_t^{-1} - P_{t-1}^{-1}||_F = ||A_t||_F = gamma/T
    # ================================================================

    # Problem setup
    T = 10
    gamma_list = np.arange(0.5, 2.5 + 1e-12, 0.5)

    # gamma used for instantaneous-cost figure
    gamma_inst = 0.5

    step_size = 0.0005
    l1_grid = np.arange(0.0, 0.5 + 1e-12, step_size)
    l3_grid = np.arange(0.0, 0.5 + 1e-12, step_size)

    theta = 30.0 * np.pi / 180.0
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=float)
    Lambda = np.diag([5.0, 1.0])
    P0 = Q @ Lambda @ Q.T

    scenarios = [
        {"name": "Scenario 1 (Benchmark)", "cost_fn": trace_fn},
        {"name": "Scenario 2", "cost_fn": offdiag_fn},
        {"name": "Scenario 3", "cost_fn": fro_fn},
    ]

    num_gamma = len(gamma_list)
    num_scen = len(scenarios)

    # Storage
    J_total = np.zeros((num_scen, num_gamma))
    trace_seq_all = [[None for _ in range(num_gamma)] for _ in range(num_scen)]
    inst_cost_all = [[None for _ in range(num_gamma)] for _ in range(num_scen)]
    P_seq_all = [[None for _ in range(num_gamma)] for _ in range(num_scen)]
    A_seq_all = [[None for _ in range(num_gamma)] for _ in range(num_scen)]
    regret_all = np.zeros((num_scen, T, num_gamma))

    # Main loop over gamma
    for ig, gamma in enumerate(gamma_list):
        tau = gamma / T
        A_grid = build_actions_fro_fixed_attention(tau, l1_grid, l3_grid)

        print("\n" + "=" * 60)
        print(f"gamma = {gamma:.2f}, per-stage attention = {tau:.4f}, |A_grid| = {len(A_grid)}")
        print("=" * 60)

        for m in range(num_scen):
            P_seq, A_seq, trace_seq, inst_cost_seq = run_myopic_rollout(
                P0, A_grid, T, scenarios[m]["cost_fn"]
            )

            P_seq_all[m][ig] = P_seq
            A_seq_all[m][ig] = A_seq
            trace_seq_all[m][ig] = trace_seq
            inst_cost_all[m][ig] = inst_cost_seq

            # Total cost is ALWAYS benchmark trace cost
            J_total[m, ig] = np.sum(trace_seq)

        # Regret relative to Scenario 1 using trace
        trace_benchmark = trace_seq_all[0][ig]
        regret_all[0, :, ig] = np.zeros(T)
        for m in range(1, num_scen):
            trace_m = trace_seq_all[m][ig]
            regret_all[m, :, ig] = (trace_m - trace_benchmark) / trace_benchmark

        # Printout for this gamma
        for m in range(num_scen):
            print(f"\n{scenarios[m]['name']}")
            print(f"   Total cost J^[{m+1}](gamma) = {J_total[m, ig]:.10f}")
            print(f"   Instantaneous costs      = {vec2str(inst_cost_all[m][ig], 10)}")
            print(f"   Trace sequence           = {vec2str(trace_seq_all[m][ig], 10)}")
            print(f"   Regret sequence          = {vec2str(regret_all[m, :, ig], 10)}")

    # Compact summary at the end
    print("\n\n==================== FINAL SUMMARY ====================")
    for ig, gamma in enumerate(gamma_list):
        print(f"\nGamma = {gamma:.2f}")
        for m in range(num_scen):
            print(f"   {scenarios[m]['name']}:")
            print(f"      Total cost     = {J_total[m, ig]:.10f}")
            print(f"      Inst. cost seq = {vec2str(inst_cost_all[m][ig], 6)}")
            print(f"      Regret seq     = {vec2str(regret_all[m, :, ig], 6)}")

    # Create output directory
    figdir = Path("Figures") / "figure_arxiv"
    figdir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Total benchmark cost J^[m](gamma)
    plt.figure(facecolor="white")
    plt.plot(gamma_list, J_total[0, :], "-o", linewidth=1.8, markersize=6,
             label="Scenario 1 (Benchmark)")
    plt.plot(gamma_list, J_total[1, :], "--s", linewidth=1.8, markersize=6,
             label="Scenario 2")
    plt.plot(gamma_list, J_total[2, :], ":d", linewidth=2.0, markersize=6,
             label="Scenario 3")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"Total benchmark cost  $J^{[m]}(\gamma)$")
    plt.title("Total benchmark cost under the Frobenius-norm attention measure")
    plt.legend(loc="best")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(figdir / "myopic_fro_total.eps", format="eps")
    plt.savefig(figdir / "myopic_fro_total.png", dpi=300)

    # Figure 2: Stagewise regret at gamma = 0.5
    idx = np.where(np.abs(gamma_list - 0.5) < 1e-12)[0]
    if len(idx) == 0:
        raise RuntimeError("gamma = 0.5 not found in gamma_list.")
    ig_regret = int(idx[0])

    t_vals = np.arange(1, T + 1)
    regret_s2 = regret_all[1, :, ig_regret]
    regret_s3 = regret_all[2, :, ig_regret]

    plt.figure(facecolor="white")
    plt.plot(t_vals, regret_s2, "--s", linewidth=1.8, markersize=6, label="Scenario 2")
    plt.plot(t_vals, regret_s3, ":d", linewidth=2.0, markersize=6, label="Scenario 3")
    plt.xlabel("Stage t")
    plt.ylabel(r"Regret$^{[m]}(0.5,t)$")
    plt.title(r"Stagewise regret at $\gamma = 0.5$ under the Frobenius-norm attention measure")
    plt.legend(loc="best")
    plt.xticks(t_vals)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(figdir / "myopic_fro_regret.eps", format="eps")
    plt.savefig(figdir / "myopic_fro_regret.png", dpi=300)

    # Figure 3: Instantaneous stage costs at gamma_inst
    idx = np.where(np.abs(gamma_list - gamma_inst) < 1e-12)[0]
    if len(idx) == 0:
        raise RuntimeError(f"gamma_inst = {gamma_inst:.4f} not found in gamma_list.")
    ig_inst = int(idx[0])

    inst_s1 = inst_cost_all[0][ig_inst]
    inst_s2 = inst_cost_all[1][ig_inst]
    inst_s3 = inst_cost_all[2][ig_inst]

    plt.figure(facecolor="white")
    plt.plot(t_vals, inst_s1, "-o", linewidth=1.8, markersize=6,
             label=r"Scenario 1: Tr($P_t^{[1]}$)")
    plt.plot(t_vals, inst_s2, "--s", linewidth=1.8, markersize=6,
             label=r"Scenario 2: $\Sigma_{i\neq j}|P_{ij}^{[2]}|$")
    plt.plot(t_vals, inst_s3, ":d", linewidth=2.0, markersize=6,
             label=r"Scenario 3: $\|P_t^{[3]}\|_F$")
    plt.xlabel("Stage t")
    plt.ylabel("Instantaneous stage cost")
    plt.title(
        rf"Instantaneous stage costs at $\gamma = {gamma_inst:.1f}$ under the Frobenius-norm attention measure"
    )
    plt.legend(loc="best")
    plt.xticks(t_vals)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(figdir / "myopic_fro_instant_cost.eps", format="eps")
    plt.savefig(figdir / "myopic_fro_instant_cost.png", dpi=300)

    # Show all figures
    plt.show()

    # Save numerical results
    scenario_names = np.array([s["name"] for s in scenarios], dtype=object)
    savemat(
        "myopic_fro_exact_match_with_printout.mat",
        {
            "T": T,
            "gamma_list": gamma_list,
            "gamma_inst": gamma_inst,
            "step_size": step_size,
            "l1_grid": l1_grid,
            "l3_grid": l3_grid,
            "P0": P0,
            "scenario_names": scenario_names,
            "J_total": J_total,
            "trace_seq_all": to_object_array_2d(trace_seq_all),
            "inst_cost_all": to_object_array_2d(inst_cost_all),
            "P_seq_all": to_object_array_2d(P_seq_all),
            "A_seq_all": to_object_array_2d(A_seq_all),
            "regret_all": regret_all,
        },
    )

    print("\nDone.")
    print(f"Saved figures in: {figdir}")
    print("Saved data file: myopic_fro_exact_match_with_printout.mat")