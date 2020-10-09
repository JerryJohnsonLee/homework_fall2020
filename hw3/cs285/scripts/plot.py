import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
from read_results import get_results_dataframe

## q1
# logdir = '../data/q1_MsPacman-v0_07-10-2020_15-00-56/events*'
# eventfile = glob.glob(logdir)[0]

# df = get_results_dataframe(eventfile)
# df.plot("Train_EnvstepsSoFar", ["Train_AverageReturn", "Train_BestReturn"])
# plt.xlabel("Training steps")
# plt.ylabel("Reward")
# plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
# plt.savefig("q1.png")

## q2
# vanilla_dqn_logdir = "../../submit/data/q2_dqn**/events*"
# eventfiles = glob.glob(vanilla_dqn_logdir, recursive=True)
# returns = []
# for efile in eventfiles:
#     returns.append(get_results_dataframe(efile)["Train_AverageReturn"].values)
# return_arr = np.array(returns)
# means = return_arr.mean(axis=0)
# stds = return_arr.std(axis=0)
# steps = get_results_dataframe(eventfiles[0])["Train_EnvstepsSoFar"].values

# plt.plot(steps, means, label="vanilla_DQN")
# plt.fill_between(steps, means - stds, means + stds,  alpha=0.3)

# ddqn_logdir = "../../submit/data/q2_doubledqn**/events*"
# eventfiles = glob.glob(ddqn_logdir, recursive=True)
# returns = []
# for efile in eventfiles:
#     returns.append(get_results_dataframe(efile)["Train_AverageReturn"].values)
# return_arr = np.array(returns)
# means = return_arr.mean(axis=0)
# stds = return_arr.std(axis=0)
# steps = get_results_dataframe(eventfiles[0])["Train_EnvstepsSoFar"].values

# plt.plot(steps, means, label="double_DQN")
# plt.fill_between(steps, means - stds, means + stds, alpha=0.3)
# plt.plot([0, 5e5], [150, 150], color="grey", alpha=0.5)

# plt.legend()
# plt.xlabel("Training steps")
# plt.ylabel("Reward")
# plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
# plt.savefig("q2.png")

# q3
# plt.figure()
# unchanged_logdir = "../../submit/data/q2_dqn_1**/events*"
# eventfile_unchanged = glob.glob(unchanged_logdir, recursive=True)[0]
# df_unchanged = get_results_dataframe(eventfile_unchanged)
# plt.plot(df_unchanged["Train_EnvstepsSoFar"], df_unchanged["Train_AverageReturn"], label="original exploration schedule")


# hp_set_1 = "../../submit/data/q3_1_0.1**/events*"
# eventfile_1 = glob.glob(hp_set_1, recursive=True)[0]
# df1 = get_results_dataframe(eventfile_1)
# plt.plot(df1["Train_EnvstepsSoFar"], df1["Train_AverageReturn"], label="exploration schedule 1")

# hp_set_2 = "../../submit/data/q3_0.5_0.1**/events*"
# eventfile_2 = glob.glob(hp_set_2, recursive=True)[0]
# df2 = get_results_dataframe(eventfile_2)
# plt.plot(df2["Train_EnvstepsSoFar"], df2["Train_AverageReturn"], label="exploration schedule 2")

# hp_set_3 = "../../submit/data/q3_0.01**/events*"
# eventfile_3 = glob.glob(hp_set_3, recursive=True)[0]
# df3 = get_results_dataframe(eventfile_3)
# plt.plot(df3["Train_EnvstepsSoFar"], df3["Train_AverageReturn"], label="exploration schedule 3")
# plt.legend()
# plt.xlabel("Training steps")
# plt.ylabel("Reward")
# plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
# plt.savefig("q3.png")

# plt.figure()
# plt.plot([0, 5e4, 5e5], [1, 0.02, 0.02], label="original exploration schedule")
# plt.plot([0, 5e4, 5e5], [1, .1, .1], label="exploration schedule 1")
# plt.plot([0, 5e4, 5e5], [0.5, 0.1, 0.1], label="exploration schedule 2")
# plt.plot([0, 5e4, 5e5], [0.01, 0.01, 0.01], label="exploration schedule 3")
# plt.legend()
# plt.xlabel("Training steps")
# plt.ylabel("$\epsilon$-value")
# plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
# plt.savefig("q3_schedule.png")

# q4
# plt.figure()
# q4_1_1_logdir = "../../submit/data/q4_ac_1_1**/events*"
# eventfile_unchanged = glob.glob(q4_1_1_logdir, recursive=True)[0]
# df_unchanged = get_results_dataframe(eventfile_unchanged)
# plt.plot(df_unchanged["Train_EnvstepsSoFar"], df_unchanged["Train_AverageReturn"], label="ntu=1, ngsptu=1")


# hp_set_1 = "../../submit/data/q4_1_100**/events*"
# eventfile_1 = glob.glob(hp_set_1, recursive=True)[0]
# df1 = get_results_dataframe(eventfile_1)
# plt.plot(df1["Train_EnvstepsSoFar"], df1["Train_AverageReturn"], label="ntu=1, ngsptu=100")

# hp_set_2 = "../../submit/data/q4_10_10**/events*"
# eventfile_2 = glob.glob(hp_set_2, recursive=True)[0]
# df2 = get_results_dataframe(eventfile_2)
# plt.plot(df2["Train_EnvstepsSoFar"], df2["Train_AverageReturn"], label="ntu=10, ngsptu=10")

# hp_set_3 = "../../submit/data/q4_100_1**/events*"
# eventfile_3 = glob.glob(hp_set_3, recursive=True)[0]
# df3 = get_results_dataframe(eventfile_3)
# plt.plot(df3["Train_EnvstepsSoFar"], df3["Train_AverageReturn"], label="ntu=100, ngsptu=1")
# plt.legend()
# plt.xlabel("Training steps")
# plt.ylabel("Reward")
# plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
# plt.savefig("q4.png")

# q5
# plt.figure()
# logdir1 = "../../submit/data/q5_1_100_HalfCheetah-v2_08-10-2020_18-24-40/events*"
# eventfile_unchanged = glob.glob(logdir1, recursive=True)[0]
# df_unchanged = get_results_dataframe(eventfile_unchanged)
# plt.plot( df_unchanged["Train_AverageReturn"])
# plt.plot([0, 150], [150, 150], color="grey", alpha=0.5)
# plt.xlabel("Number of iterations")
# plt.ylabel("Reward")
# plt.title("HalfChettah")
# plt.savefig("q5-halfchettah.png")

plt.figure()
logdir2 = "../../submit/data/q5_1_100_InvertedPendulum-v2_08-10-2020_18-18-00/events*"
eventfile_unchanged = glob.glob(logdir2, recursive=True)[0]
df_unchanged = get_results_dataframe(eventfile_unchanged)
plt.plot(np.arange(0,100, 10), df_unchanged["Train_AverageReturn"])
plt.plot([0, 100], [1000, 1000], color="grey", alpha=0.5)
plt.xlabel("Number of iterations")
plt.ylabel("Reward")
plt.title("InvertedPendulum")
plt.savefig("q5-invertedpendulum.png")
