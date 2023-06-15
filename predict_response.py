# Bibliotecas de pré-processamento de dados de texto
import nltk

import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]


# Biblioteca load_model
import tensorflow 
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model('chatbot_model.h5')

# Carregue os arquivos de dados
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#função recebe a entrada do usuário, a tokeniza e converte o texto tokenizado em palavras-tronco.

def preprocess_user_input(user_input):

    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
    # sorted, organização de lista
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []
   
    # Codificação dos dados de entrada 
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
  
    return np.array(bag)

##função que irá receber o texto do usuário

def bot_class_prediction(user_input):

    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label


#uma função para dar uma resposta de acordocom a entrada do usuário
def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
   predicted_class = classes[predicted_class_label]

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
        bot_response = random.choice(intent['responses'])
        return bot_response

print("Oi, eu sou a Estela, como posso ajudar?")

while True:
    user_input = input("Digite sua mensagem aqui:")
    print("Entrada do Usuário: ", user_input)

    response = bot_response(user_input)
    print("Resposta do Robô: ", response)

