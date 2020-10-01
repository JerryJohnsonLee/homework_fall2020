import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from read_results import get_section_results
from glob import glob


# _, return_lb_no_rtg_dsa = get_section_results(glob("../../submit/data/q1_sb_no_rtg_dsa_CartPole-v0_26-09-2020_12-19-55/*")[0])
# _, return_lb_rtg_dsa = get_section_results(glob("../../submit/data/q1_sb_rtg_dsa_CartPole-v0_26-09-2020_12-23-55/*")[0])
# _, return_lb_rtg_na = get_section_results(glob("../../submit/data/q1_sb_rtg_na_CartPole-v0_26-09-2020_12-22-38/*")[0])

# plt.plot(iters, return_lb_no_rtg_dsa, label="no_rtg_dsa")
# plt.plot(iters, return_lb_rtg_dsa, label="rtg_dsa")
# plt.plot(iters, return_lb_rtg_na, label="rtg_na")

# _,returns = get_section_results(glob("../../submit/data/q3_b40000_r0.005_LunarLanderContinuous-v2_21-09-2020_21-28-16/*")[0])

iters = np.arange(100) + 1
# for b in ["10000", "30000", "50000"]:
#     for lr in ["0.005", "0.01", "0.02"]:
#         _, returns = get_section_results(glob(f"../../submit/data/q4_search_b{b}_lr{lr}_*/*")[0])
#         plt.plot(iters, returns, label=f"b={b}, lr={lr}")


# _, return__ = get_section_results(glob("../../submit/data/q4_b30000_r0.02_HalfCheetah-v2_25-09-2020_13-40-22/*")[0])
# _, return_baseline = get_section_results(glob("../../submit/data/q4_b30000_r0.02_nnbaseline_HalfCheetah-v2_25-09-2020_14-33-56/*")[0])
# _, return_rtg = get_section_results(glob("../../submit/data/q4_b30000_r0.02_rtg_HalfCheetah-v2_25-09-2020_14-07-56/*")[0])
# _, return_all = get_section_results(glob("../../submit/data/q4_b30000_r0.02_rtg_nnbaseline_HalfCheetah-v2_25-09-2020_15-12-22/*")[0])

# plt.plot(iters, return__, label="vanilla")
# plt.plot(iters, return_baseline, label="with baseline")
# plt.plot(iters, return_rtg, label="with reward-to-go")
# plt.plot(iters, return_all, label="with baseline and reward-to-go")

sequential, _ = get_section_results(glob("../../submit/data/q3_b40000_r0.005_LunarLanderContinuous-v2_21-09-2020_21-28-16/*")[0])
parallel, _ = get_section_results(glob("../../submit/data/q3_b40000_r0.005_parallel8_LunarLanderContinuous-v2_24-09-2020_23-18-37/*")[0])

plt.plot(iters, np.array(sequential)/60, label="sequential")
plt.plot(iters, np.array(parallel)/60, label="parallel")

plt.xlabel("Iterations")
plt.ylabel("Traning time (min)")
plt.legend()
plt.title("Training time for LunarLander control\n with different trajectory collecting methods")
plt.savefig("bonus2.png")