from tensorflow.keras import applications, layers, models, optimizers

base = applications.MobileNetV2(weights='imagenet', include_top=False,
                                input_shape=(160, 160, 3))

inputs = layers.Input((160, 160, 3))
x = base(inputs, training=False)              # BatchNorm en inférence
x = layers.GlobalAveragePooling2D()(x)
model = models.Model(inputs, layers.Dense(1, activation='sigmoid')(x))

# Phase 1 : Feature Extraction
base.trainable = False
model.compile(optimizer=optimizers.Adam(1e-3),
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Phase 2 : Fine-Tuning
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=optimizers.Adam(1e-5),   # recompiler !
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)