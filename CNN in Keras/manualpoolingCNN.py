# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])


#Convolution -> Max pooling -> Convolution -> Flatten -> Dense
# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu',
                input_shape=(img_rows, img_cols, 1)))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

# Fit to training data
model.fit(train_data, train_labels, validation_split=0.2,
epochs=3, batch_size=10)

# Evaluate on test data
model.evaluate(test_data, test_labels, batch_size=10)
