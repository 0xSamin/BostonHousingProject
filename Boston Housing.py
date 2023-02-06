import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from pandas import read_csv
from sklearn.model_selection import train_test_split

# from tensorflow.keras import layers


# loading the dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(
    'https://storage.googleapis.com/kagglesdsdata/datasets/1815/3139/housing.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221227%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221227T080001Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=31a23e4a09e79ae879aeaa551f9f9e016ae6d1e4bf07d8dd7039c6ba7da4e9c5cc701a37fdb12f1c2bd9ae6ffa3f8c2b6fcc7db852a2732cde7b1af6b9c00e8bc0e5424576e05fff77ce4364a3c02f2b3f333aa435c51bab50585cbf7005f14d80362167fc39bd1143e239481ad0a2f44f1bb3bcd99d37dd2824938fe9c3019f5fa2958109faabaa853524cee64fed621d5b7b770aad074383e41878ffa1cd5b311b6545c33537faca83c38f77dc9ab388f9f539ccf139df8984be734d83f36d51c1366b29a7b1b184cecc604be59467a8769b72e29ee6b197d0eb6c74e2a64d7b67f44abf0f2e3447b63ebac0dd29a006f8343b0c3ffeb1e4510dd6970e78e8',
    header=None, delimiter=r"\s+", names=column_names)

# print(data.head(5))
# print(np.shape(data))


# splitting data
y = data.MEDV  # this is the target
x = data.drop('MEDV', axis=1)

X_train, X_test, train_targets, test_targets = train_test_split(x, y, test_size=0.2)
"""print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

# returning targets
print(train_targets)"""

# Normalizing the data
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std
X_test -= mean
X_test /= std


# Model Defination
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# choosing loss function
# K-Fold Validation
k = 4
num_val_samples = len(X_train) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #%d' % i)
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]  # 1
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(  # 2
        [X_train[:i * num_val_samples],
         X_train[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()  # 3
    model.fit(partial_train_data, partial_train_targets,  # 4
              epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)  # 5
    all_scores.append(val_mae)

# Validating approach using K-fold validation
print(all_scores, np.mean(all_scores))

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #%d' % i)
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([X_train[:i * num_val_samples], X_train[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

model = build_model()
model.fit(X_train, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(X_train, test_targets)

predictions = model.predict(X_train)
