import matplotlib.pyplot as plt
import numpy as np

iters = np.arange(10)
# avg_rtn = np.array([1724.2, 4248.7, 4562.3, 4502.7, 4634.6, 4731.8, 4797.9, 4626.8, 4757.7, 4810.2])
# std_rtn = np.array([621.7, 176.8, 79.4, 66, 182.4, 106.1, 49.8, 94.9, 11.7, 93.8])
# bc_baseline = 1724.2
# expert = 4713.7

avg_rtn = np.array([266.9, 255.1, 344.9, 299.6, 275.9, 340, 326, 320, 357.2, 406.8])
std_rtn = np.array([38.7, 12.2, 78.9, 15.3, 37.1, 34.9, 43.3, 32.8, 80.4, 113.9])
bc_baseline = 266.9
expert = 10345

plt.errorbar(iters, avg_rtn, std_rtn, ecolor="grey", capsize=3, label="learning curve for DAgger")
plt.plot([0, max(iters)], [bc_baseline, bc_baseline], "k:", label="behavioral cloning agent performance")
plt.plot([0, max(iters)], [expert, expert], "k", label="expert policy performance")

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Evaluation performance")
plt.title("Environment: Humanoid-v2")
plt.show()