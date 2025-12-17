import tensorflow as tf
from scripts.data_collection import download_dataset

def create_data_pipeline(ds_train, ds_test, batch_size=32, image_size=(224, 224)):
    """
    Create data pipeline with augmentation.
    """
    def preprocess_image(image, label):
        # Resize
        image = tf.image.resize(image, image_size)
        # Normalize to [0,1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def augment_image(image, label):
        # Random augmentations
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.1)
        return image, label
    
    # Apply preprocessing
    ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation to training
    ds_train = ds_train.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test

if __name__ == "__main__":
    ds_train, ds_test = download_dataset()
    train_ds, test_ds = create_data_pipeline(ds_train, ds_test)
    print("Data pipeline created.")
    print(f"Train batches: {len(list(train_ds))}")
    print(f"Test batches: {len(list(test_ds))}")