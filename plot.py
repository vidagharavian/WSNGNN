from matplotlib import pyplot as plt
import numpy as np

npload=np.load("hist_train_test.npz")
hist_train_loss=npload['hist_train_loss']
hist_test_loss=npload['hist_test_loss']
hist_train_f1=npload['hist_train_f1']
hist_test_f1=npload['hist_test_f1']

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(hist_train_loss, label="Train")
ax.plot(hist_test_loss, label="Test")
ax.set_title("Loss per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.savefig("plot_loss.png")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(hist_train_f1, label="Train")
ax.plot(hist_test_f1, label="Test")
ax.set_title("F1-score per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1-score")
ax.legend()
plt.savefig("plot_f1.png")