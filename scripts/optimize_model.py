import tensorflow as tf
import os

def convert_to_tflite(model_path, output_path):
    """
    Convert Keras model to TensorFlow Lite with quantization.
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    
    # Get model size
    model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Model size: {model_size:.2f} MB")
    
    return tflite_model

def validate_accuracy(tflite_model, test_ds):
    """
    Validate accuracy drop after quantization.
    """
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    total = 0
    
    for images, labels in test_ds:
        for i in range(images.shape[0]):
            # Preprocess
            input_data = tf.expand_dims(images[i], 0)
            input_data = tf.cast(input_data, tf.float32)
            
            # Set input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output_data)
            
            if pred == labels[i].numpy():
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"TFLite model accuracy: {accuracy:.4f}")
    
    return accuracy

if __name__ == "__main__":
    # Convert fine-tuned model
    tflite_model = convert_to_tflite('models/fine_tuned_model.h5', 'models/model.tflite')
    
    # For validation, need test_ds
    from scripts.data_pipeline import create_data_pipeline
    from scripts.data_collection import download_dataset
    ds_train, ds_test = download_dataset()
    _, test_ds = create_data_pipeline(ds_train, ds_test)
    
    validate_accuracy(tflite_model, test_ds)