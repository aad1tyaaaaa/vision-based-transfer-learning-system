import tensorflow as tf
from scripts.train_baseline import create_model
from scripts.data_pipeline import create_data_pipeline
from scripts.data_collection import download_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

def fine_tune_model():
    """
    Fine-tune the model by unfreezing some layers.
    """
    # Load baseline model
    model = tf.keras.models.load_model('models/baseline_model.h5')
    
    # Unfreeze the last few layers of base model
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Get data
    ds_train, ds_test = download_dataset()
    train_ds, test_ds = create_data_pipeline(ds_train, ds_test)
    
    # Fine-tune
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=5
    )
    
    # Save fine-tuned model
    model.save('models/fine_tuned_model.h5')
    
    return model, history

def evaluate_model(model, test_ds):
    """
    Evaluate the model with confusion matrix and metrics.
    """
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=['Cat', 'Dog'])
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Cat', 'Dog'])
    plt.yticks(tick_marks, ['Cat', 'Dog'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('docs/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    model, history = fine_tune_model()
    # For evaluation, need test_ds
    ds_train, ds_test = download_dataset()
    _, test_ds = create_data_pipeline(ds_train, ds_test)
    evaluate_model(model, test_ds)
    print("Fine-tuning completed and evaluated.")