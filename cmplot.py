import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

npload=np.load("Result4/y_pred_true.npz")
y_pred=npload['y_pred']
y_true=npload['y_true']
print(y_pred.shape, y_true.shape)
target_names=npload['target_names']

print(classification_report(y_true, y_pred, target_names=target_names, digits=2, output_dict=True))

cm = confusion_matrix(y_true, y_pred, labels=np.arange(0,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=target_names)
disp.plot()

plt.savefig("Result4/confusion_matrix.png")