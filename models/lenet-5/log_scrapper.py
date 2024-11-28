import re

# Inicjalizacja list
epochs = [1,2,3,4,5,6,7,8,9,10]

train_loss = [0.685, 0.320, 0.315, 0.291, 0.292, 0.291, 0.294, 0.300, 0.339, 0.484]
test_loss = [52.525, 79.907, 44.754, 34.964, 23.567, 24.818, 20.611, 20.019, 20.405, 16.721]

train_acc_hardness = [0.966, 0.968, 0.956, 0.967, 0.965, 0.966, 0.968, 0.957, 0.961, 0.953]
train_acc_honey = [0.993, 0.994, 0.994, 0.993, 0.993, 0.993, 0.992, 0.989, 0.983, 0.973]
train_acc_capped = [0.986, 0.982, 0.982, 0.976, 0.983, 0.983, 0.978,  0.974, 0.967, 0.959]

test_acc_hardness = [0.930,0.930,0.930,0.930,0.930,0.930,0.930,0.930,0.930,0.930]
test_acc_honey = [0.209,0.209,0.209,0.209,0.209,0.209,0.209,0.209,0.209,0.209]
test_acc_capped = [0.140,0.140,0.140,0.140,0.140,0.140,0.140,0.140,0.140,0.140]

train_bal_acc_hardness = [0.964, 0.963, 0.948, 0.962, 0.960, 0.960, 0.962, 0.949, 0.954, 0.946]
train_bal_acc_honey = [0.993,  0.993, 0.994, 0.992, 0.992, 0.992, 0.992, 0.988, 0.983, 0.971]
train_bal_acc_capped = [0.976, 0.972, 0.974, 0.969, 0.971, 0.971, 0.971, 0.969, 0.963, 0.954]

test_bal_acc_hardness = [0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500]
test_bal_acc_honey = [0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500]
test_bal_acc_capped = [0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500]

print(len(epochs), len(train_loss), len(test_loss), len(train_acc_hardness), len(train_acc_honey), len(train_acc_capped), len(test_acc_hardness), len(test_acc_honey), len(test_acc_capped), len(train_bal_acc_hardness), len(train_bal_acc_honey), len(train_bal_acc_capped), len(test_bal_acc_hardness), len(test_bal_acc_honey), len(test_bal_acc_capped))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, test_loss, label='Test Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss in function of epochs, first fold')
plt.legend()
plt.grid(True)
plt.savefig('./loss.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc_hardness, label='Train Accuracy - Hardness', marker='o')
plt.plot(epochs, train_acc_honey, label='Train Accuracy - Honey', marker='o')
plt.plot(epochs, train_acc_capped, label='Train Accuracy - Capped', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.title('Train data accuracy in function of epochs, first fold')
plt.legend()
plt.grid(True)
plt.savefig('./train_acc.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, test_acc_hardness, label='Test Accuracy - Hardness', marker='o')
plt.plot(epochs, test_acc_honey, label='Test Accuracy - Honey', marker='o')
plt.plot(epochs, test_acc_capped, label='Test Accuracy - Capped', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test data accuracy in function of epochs, first fold')
plt.legend()
plt.grid(True)
plt.savefig('./test_acc.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, test_bal_acc_hardness, label='Test Balanced Accuracy - Hardness', marker='o')
plt.plot(epochs, test_bal_acc_honey, label='Test Balanced Accuracy - Honey', marker='o')
plt.plot(epochs, test_bal_acc_capped, label='Test Balanced Accuracy - Capped', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')
plt.title('Test Balanced Accuracy in function of epochs, first fold')
plt.legend()
plt.grid(True)
plt.savefig('./balanced_test_acc.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_bal_acc_hardness, label='Train Balanced Accuracy - Hardness', marker='o')
plt.plot(epochs, train_bal_acc_honey, label='Train Balanced Accuracy - Honey', marker='o')
plt.plot(epochs, train_bal_acc_capped, label='Train Balanced Accuracy - Capped', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')
plt.title('Train Balanced Accuracy in function of epochs, first fold')
plt.legend()
plt.grid(True)
plt.savefig('./balanced_train_acc.png')
