***** Replace the SVM classifier definition with this in the extraction.py script*****



model = models.Sequential()

# Input Layer
hidden_size = 768  # dimensional embeddings
model.add(layers.InputLayer(input_shape=(hidden_size,)))

# Hidden Layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))  # Dropout to prevent overfitting
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))

# Output Layer (Number of classes)
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model to understand its architecture
model.summary()
