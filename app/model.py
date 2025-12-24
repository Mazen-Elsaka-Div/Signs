from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def build_model(input_shape, num_classes):
    """
    Build a Dense network for hand landmark classification
    Input: (63,) - 21 landmarks × 3 coordinates (x, y, z)
    Output: num_classes probabilities
    
    This is NOT a CNN - it's a simple Dense network for landmarks!
    """
    classes = num_classes if num_classes > 0 else 1
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First dense block
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second dense block
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third dense block
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Fourth dense block
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(classes, activation='softmax')
    ])
    
    # Compile with optimizer
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model