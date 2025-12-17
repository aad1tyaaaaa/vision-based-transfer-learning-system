import tensorflow as tf
from scripts.data_pipeline import create_data_pipeline
from scripts.data_collection import download_dataset
import os

def create_model(num_classes=2):
    """
    Create transfer learning model with MobileNetV2.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """
    Train the baseline model.
    """
    # Get data
    ds_train, ds_test = download_dataset()
    train_ds, test_ds = create_data_pipeline(ds_train, ds_test)
    
    # Create model
    model = create_model()
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=5  # Small number for baseline
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/baseline_model.h5')
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()
    print("Baseline model trained and saved.")