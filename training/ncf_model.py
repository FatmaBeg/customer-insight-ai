from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam

def create_ncf_model(num_users, num_categories, embed_dim=16):
    user_input = Input(shape=(1,))
    category_input = Input(shape=(1,))
    
    user_embed = Embedding(input_dim=num_users, output_dim=embed_dim)(user_input)
    category_embed = Embedding(input_dim=num_categories, output_dim=embed_dim)(category_input)
    
    user_vec = Flatten()(user_embed)
    category_vec = Flatten()(category_embed)
    
    concat = Concatenate()([user_vec, category_vec])
    
    x = Dense(128, activation='relu')(concat)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    # 🔽 İşte burada learning rate ayarı yapılıyor
    optimizer = Adam(learning_rate=0.0005)

    model = Model(inputs=[user_input, category_input], outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model