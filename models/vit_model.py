import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras as keras

class VisionTransformer(keras.Model):
    def __init__(self, patch_size, image_size, embedding_dim=64, num_heads=4, num_classes=10,
                 in_channels=1, dropout_rate=0.1, num_blocks=3):
        super(VisionTransformer, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        # Patch embedding with proper in_channels
        self.patch_embedding = layers.Conv2D(embedding_dim, kernel_size=patch_size,
                                           strides=patch_size, padding='valid')

        # Fixed positional encoding using Embedding layer
        max_positions = (image_size // patch_size) ** 2
        self.position_embedding = layers.Embedding(max_positions, embedding_dim)

        # Initial dropout
        self.dropout = layers.Dropout(dropout_rate)
        
        # Multiple transformer blocks
        self.transformer_blocks = []
        for i in range(num_blocks):
            block = {
                'layer_norm1': layers.LayerNormalization(epsilon=1e-6),
                'attention': layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embedding_dim // num_heads,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42)
                ),
                'layer_norm2': layers.LayerNormalization(epsilon=1e-6),
                'ffn': keras.Sequential([
                    layers.Dense(embedding_dim * 2, activation='relu'),  # 4x expansion
                    layers.Dropout(dropout_rate),
                    layers.Dense(embedding_dim)
                ]),
            }
            self.transformer_blocks.append(block)

        # Global pooling and classification
        self.global_avg_pool = tf.keras.layers.Flatten()
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, x, return_attention=False, training=False):
        # Patch embedding (unchanged)
        patches = self.patch_embedding(x)
        
        batch_size = tf.shape(patches)[0]
        h = tf.shape(patches)[1]
        w = tf.shape(patches)[2]
        num_patches = h * w
        
        patches = tf.reshape(patches, [batch_size, num_patches, self.embedding_dim])
        
        # Add positional encoding (unchanged)
        positions = tf.range(num_patches)
        pos_encoding = self.position_embedding(positions)
        x = patches + pos_encoding
        x = self.dropout(x, training=training)
        
        attention_maps_to_return = None
        
        # Process through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Multi-head attention with residual connection
            residual = x
            x = block['layer_norm1'](x)
            
            # Attention block
            if i == 0 and return_attention:
                attn_output, attention_maps = block['attention'](
                    query=x, key=x, value=x, 
                    return_attention_scores=True,
                    training=training
                )
                attention_maps_to_return = attention_maps
            else:
                attn_output = block['attention'](x, x, x, training=training)
            
            # REMOVED: x = block['attention_dropout'](attn_output, training=training)
            x = attn_output + residual  # Residual connection
            
            # Feed-forward network
            residual = x
            x = block['layer_norm2'](x)
            x = block['ffn'](x, training=training)  # Dropout is inside the FFN
            # REMOVED: x = block['ffn_dropout'](x, training=training)
            x = x + residual  # Residual connection
        
        # Classification head (unchanged)
        x = self.global_avg_pool(x)
        output = self.classifier(x)
        
        if return_attention and attention_maps_to_return is not None:
            return output, attention_maps_to_return
        else:
            return output