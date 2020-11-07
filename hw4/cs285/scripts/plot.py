import matplotlib
from matplotlib.pyplot import legend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
from read_results import get_results_dataframe

## q3
# logdir = '../../data/hw4_q3_cheetah_cheetah-cs285-v0_31-10-2020_14-29-41/events*'
# eventfile = glob.glob(logdir)[0]

# df = get_results_dataframe(eventfile)
# df.plot(y=["Train_AverageReturn", "Eval_AverageReturn"])
# plt.scatter([0], [df["Train_AverageReturn"].iloc[0]])
# plt.scatter([0], [df["Eval_AverageReturn"].iloc[0]])
# plt.fill_between([0, 19], [250], [350], color="grey", alpha=0.5)
# plt.xlabel("Number of training iterations")
# plt.ylabel("Return")
# plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
# plt.title("Cheetach")
# plt.savefig("q3_3.png")

# q4
logdir = '../../data/hw4_q4_*/events*'
eventfiles = sorted(glob.glob(logdir, recursive=True))
for eventfile in eventfiles:
    name = eventfile.split("_")[3]
    df = get_results_dataframe(eventfile)
    plt.plot(df.index, df["Eval_AverageReturn"], label=name)

plt.plot([0, 14], [-300, -300], ":", color="grey", alpha=0.5)
plt.xlabel("Number of training iterations")
plt.ylabel("Return")
plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
plt.legend()
plt.savefig("q4.png")

