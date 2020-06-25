import numpy as np
import tensorflow as tf

npz = np.load('Audiobooks_data_train.npz')
train_inputs, train_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

input_size = 10
output_size = 2
hidden_layer_size = 300

# Define the amount of layers (width and height of the model) and the activation functions to be used per layer 
model = tf.keras.Sequential([
                          
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                        
                            tf.keras.layers.Dense(output_size, activation='softmax')
                            ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Traning and fitting the model
batch_size = 100
num_epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(train_inputs,
          train_targets,
          batch_size=batch_size,
          epochs=num_epochs,
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose=2)

#Testing the model
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))