import time
import os
import numpy as np
from PIL import Image
import tensorflow as tf

def benchmark_model(model_path, test_images):
    """
    Benchmark model size and latency.
    """
    # Model size
    if model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"Model size: {model_size:.2f} MB")
    elif model_path.endswith('.tflite'):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"TFLite model size: {model_size:.2f} MB")
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Latency
        latencies = []
        for img_path in test_images[:10]:  # Test on 10 images
            image = Image.open(img_path).convert('RGB').resize((224, 224))
            input_data = np.array(image, dtype=np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        avg_latency = np.mean(latencies) * 1000  # ms
        print(f"Average inference latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    # Example usage
    # benchmark_model('models/fine_tuned_model.h5', ['path/to/test/image1.jpg', ...])
    # benchmark_model('models/model.tflite', ['path/to/test/image1.jpg', ...])
    print("Benchmark script ready. Uncomment and provide test image paths to run.")