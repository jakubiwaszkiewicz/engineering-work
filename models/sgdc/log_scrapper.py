fold = [1,2,3,4,5]

test_bal_acc_hardness = [0.325, 0.323, 0.324, 0.413, 0.308]
test_bal_acc_honey = [0.654, 0.731, 0.697, 0.648, 0.638] 
test_bal_acc_capped = [0.725, 0.571, 0.696, 0.587, 0.572]

import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.plot(fold, test_bal_acc_hardness, label='Test Balanced Accuracy - Hardness', marker='o')
plt.plot(fold, test_bal_acc_honey, label='Test Balanced Accuracy - Honey', marker='o')
plt.plot(fold, test_bal_acc_capped, label='Test Balanced Accuracy - Capped', marker='o')
plt.xlabel('Fold')
plt.ylabel('Balanced Accuracy')
plt.title('Test Balanced Accuracy in function of folds')
plt.legend()
plt.grid(True)
plt.savefig('./balanced_test_acc.png')
