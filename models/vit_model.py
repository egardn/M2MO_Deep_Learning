import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras as keras

class VisionTransformer(keras.Model):
    def __init__(self, patch_size, image_size, embedding_dim=64, num_heads=4, num_classes=8,
                 in_channels=1, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # Patch embedding with smaller embedding dimension
        self.patch_embedding = layers.Conv2D(embedding_dim, kernel_size=patch_size,
                                           strides=patch_size, padding='valid')

        # Fixed positional encoding using Embedding layer instead of learned weights
        max_positions = (image_size // patch_size) ** 2
        self.position_embedding = layers.Embedding(max_positions, embedding_dim)

        # Dropout layers
        self.dropout = layers.Dropout(dropout_rate)
        self.attention_dropout = layers.Dropout(dropout_rate)
        self.ffn_dropout = layers.Dropout(dropout_rate)

        # Transformer layers
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42)
        )

        # LayerScale parameters - initialized to small values
        self.layer_scale1 = self.add_weight(
            name="layer_scale1",
            shape=(embedding_dim,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )
        self.layer_scale2 = self.add_weight(
            name="layer_scale2",
            shape=(embedding_dim,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )

        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # FFN with no expansion factor and ReLU activation
        self.ffn = keras.Sequential([
            layers.Dense(embedding_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embedding_dim)
        ])

        # Global pooling and classification
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes)

    def call(self, x, return_attention=False, training=False):
        # Patch embedding
        patches = self.patch_embedding(x)

        batch_size = tf.shape(patches)[0]
        h = tf.shape(patches)[1]
        w = tf.shape(patches)[2]
        num_patches = h * w

        # Reshape to sequence format [B, H, W, C] -> [B, H*W, C]
        patches = tf.reshape(patches, [batch_size, num_patches, self.embedding_dim])

        # Add positional encoding using fixed embedding
        positions = tf.range(num_patches)
        pos_encoding = self.position_embedding(positions)
        x = patches + pos_encoding
        x = self.dropout(x, training=training)

        # Transformer block with residual connections
        residual = x
        x = self.layer_norm1(x)

        # Multi-head attention with attention scores if requested
        if return_attention:
            attn_output, attention_maps = self.attention(
                query=x, key=x, value=x,
                return_attention_scores=True,
                training=training
            )
        else:
            attn_output = self.attention(x, x, x, training=training)

        # Apply attention dropout and LayerScale before residual connection
        x = self.attention_dropout(attn_output, training=training)
        x = x * self.layer_scale1[tf.newaxis, tf.newaxis, :]  # Apply LayerScale
        x = x + residual  # Residual connection

        # FFN block with residual connections
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x, training=training)
        x = self.ffn_dropout(x, training=training)
        x = x * self.layer_scale2[tf.newaxis, tf.newaxis, :]  # Apply LayerScale
        x = x + residual  # Residual connection

        # Global pooling
        x = self.global_avg_pool(x)  # [B, C]

        # Classification
        output = self.classifier(x)

        if return_attention:
            return output, attention_maps
        else:
            return output