# Bibliotecas de treinamento do modelo
##sequential para o modelo CNN 
from tensorflow.keras.models import Sequential
#import das camandas de processamento de dados
from tensorflow.keras.layers import Dense, Activation, Dropout
#import de otimização para reduzir perdas= ADAM
from tensorflow.keras.optimizers import Adam



from data_preprocessing import preprocess_train_data

def train_bot_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile o modelo
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

    # Ajuste e salve o modelo
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)
    model.save('chatbot_model.h5', history)
    print("Modelo Criado e Salvo")


# Chamando os métodos para treinar o modelo
train_x, train_y = preprocess_train_data()

train_bot_model(train_x, train_y)

##10. após processar o modelo criar um arquivo predict_response

