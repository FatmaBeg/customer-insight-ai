import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

def weighted_binary_crossentropy(pos_weight: float):
    """
    Create a weighted binary crossentropy loss function.
    
    Args:
        pos_weight: Weight to apply to the positive class (label=1)
    
    Returns:
        A function that computes weighted BCE
    """
    def loss(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Compute the weighted loss
        loss = - (pos_weight * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(loss, axis=-1)
    
    return loss