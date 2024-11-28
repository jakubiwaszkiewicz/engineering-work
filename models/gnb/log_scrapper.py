fold = [1,2,3,4,5]

test_bal_acc_hardness = [0.428, 0.395, 0.512, 0.390, 0.342]
test_bal_acc_honey = [0.766, 0.764, 0.796, 0.780, 0.791] 
test_bal_acc_capped = [0.917, 0.861, 0.795, 0.945, 0.857]

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
