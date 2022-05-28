#import os
#os.system('source /home/pi/Nagaraj/MasterProgram/venv/bin/activate')
#from d"
#cmd = 'source /home/pi/Nagaraj/MasterProgram/venv/bin/activate'
#os.system(cmd)

import glob
import os
import pickle
import pathlib
#import string
import nltk

#from os import listdir

from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import pickle as pk
#%matplotlib inline


from tqdm import tqdm
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
#from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Input, Dropout ,Embedding
#from keras.optimizers import Adam, RMSprop
#from keras.layers.wrappers import Bidirectional
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.layers.merge import add
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model
import tensorflow as tf

import time
from nltk.translate.bleu_score import corpus_bleu

# Text to Audio Google API
from gtts import gTTS
from IPython import display
from pygame import mixer


#pip install -q tensorflow-model-optimization

#from google.colab import drive
#drive.mount('/content/drive')

RASP_BERRY_PATH = "/home/nag/Nagaraj/Master"

#token = '/content/drive/MyDrive/MasterCode/Flickr8k_text/Flickr8k.token.txt'

token = RASP_BERRY_PATH+'/Flickr8k_text/Flickr8k.token.txt'


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

# extract features from each photo in the directory
def extract_features(model,file,ts=(299, 299)):
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(file, target_size=ts)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


def extract_features_quantized(extractor,file,ts=(299, 299)):
    image = load_img(file, target_size=ts)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    input_index = extractor.get_input_details()[0]["index"]
    output_index = extractor.get_output_details()[0]["index"]

    extractor.set_tensor(input_index, image)
    feature = extractor.invoke()
    feature = extractor.get_tensor(output_index)[0]
    feature = feature.reshape(1,-1)
    return feature


def extract_final_model_quantized(final_extractor,feature_output,caps_out):
    input_index_0 = final_extractor.get_input_details()[0]["index"]
    input_index_1 = final_extractor.get_input_details()[1]["index"]
    output_index = final_extractor.get_output_details()[0]["index"]
    caps_out_float = caps_out.astype(np.float32)
    final_extractor.set_tensor(input_index_0, caps_out_float)
    final_extractor.set_tensor(input_index_1, feature_output)

    final_description = final_extractor.invoke()
    final_description = final_extractor.get_tensor(output_index)[0]
    final_description = final_description.reshape(1,-1)
    return final_description


def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp


def predict_captions(image):
    print('predict_captions')
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encoding_test[image[len(images):]]
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])


def predict_captions_interpreter_encoder(image,interpreter):
    print('predict_captions_interpreter_encoder')
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        #e = encoding_test[image[len(images):]]
        output = extract_features_quantized(interpreter,image)
        #print(output)
        #print(output.size)
        
        preds = final_model.predict([output, np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        #print(word_pred,"\t")
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])


#testfeature = extract_features(final_model,im)

def predict_captions_interpreter_encoder_full(image,encoder_interpreter,decoder_interpreter,yes_encoder=True):
    print('predict_captions_interpreter_encoder_full')
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        #e = encoding_test[image[len(images):]]
        if yes_encoder == True:
          feature_output = extract_features_quantized(encoder_interpreter,image)
          preds = extract_final_model_quantized(decoder_interpreter,feature_output,np.array(par_caps))
        else:
          feature_output = encoding_test[image[len(images):]]
          preds = extract_final_model_quantized(decoder_interpreter,np.array([feature_output]),np.array(par_caps))
          #print(preds)
        #print(output.size)
  
        #preds = final_model.predict([output, np.array(par_caps)])
        
        word_pred = idx2word[np.argmax(preds[0])]
        #print(word_pred,"\t")
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

# evaluate the skill of the model
def evaluate_model_basic_oneimage(img_path,img_str_key,encoder_final_quant=0,encoder_interpreter=False,decoder_interpreter=False):
    print('evaluate_model_basic_oneimage')
    actual, predicted = list(), list()
    start = time.time()
    if encoder_final_quant == 1:
        print("\n ****** Encoder Only Quantization: ") 
        yhat = predict_captions_interpreter_encoder(img_path,encoder_interpreter)
    elif encoder_final_quant == 2:
        print("\n ****** No Encoder Quantization and Final Model Quantization: ")
        yhat = predict_captions_interpreter_encoder_full(img_path,encoder_interpreter,decoder_interpreter,False)
    elif encoder_final_quant == 3:
        print("\n ****** Both Encoder and Final Model Quantization: ")
        yhat = predict_captions_interpreter_encoder_full(img_path,encoder_interpreter,decoder_interpreter,True)
    else:
        print("\n ******NO Model Quantization: ")
        yhat = predict_captions(img_path)

    print("\n****** Time taken: ", (time.time()-start)/60)  

    references = [desp.split() for desp in d[img_str_key]]
    actual.append(references)
    predicted.append(yhat.split())
    print('ACTUAL- TEXT:')
    print(d[img_str_key])
    #predicted_text = ' '.join(predicted[1:-1]
    print('PREDICTED- TEXT:')
    print(yhat)

    tts = gTTS(yhat,slow = False)
    tts.save('pred_caption.mp3')
    
    sound_file = 'pred_caption.mp3'
    #display.display(display.Audio('/home/nag/Nagaraj/Master/pred_caption.mp3'))
    mixer.init()
    sound = mixer.music.load("pred_caption.mp3")
    mixer.music.play()
    while mixer.music.get_busy() == True:
        continue
    
    #sound.play()
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def evaluate_model_basic_oneimage_unseen_image(img_path,encoder_final_quant=0,encoder_interpreter=False,decoder_interpreter=False):
    print('evaluate_model_basic_oneimage_unseen_image')
    actual, predicted = list(), list()
    start = time.time()
    if encoder_final_quant == 1:
        print("\n ****** Encoder Only Quantization: ") 
        yhat = predict_captions_interpreter_encoder(img_path,encoder_interpreter)
    elif encoder_final_quant == 2:
        print("\n ******  No Encoder Quantization and Final Model Quantization: ")
        #yhat = predict_captions_interpreter_encoder_full(img_path,encoder_interpreter,decoder_interpreter,False)
    elif encoder_final_quant == 3:
        print("\n ****** Both Encoder and Final Model Quantization: ")
        yhat = predict_captions_interpreter_encoder_full(img_path,encoder_interpreter,decoder_interpreter,True)
    else:
        print("\n ******NOT VALID ******** ")
        #yhat = predict_captions(img_path)

    print("\n****** Time taken: ", (time.time()-start)/60)  

    #references = [desp.split() for desp in d[img_str_key]]
    #actual.append(references)
    predicted.append(yhat.split())
    #print('ACTUAL- TEXT:')
    #print(d[img_str_key])
    #predicted_text = ' '.join(predicted[1:-1])
    print('PREDICTED- TEXT:')
    print(yhat)

    tts = gTTS(yhat,slow = False)
    tts.save('pred_caption.mp3')
    
    sound_file = 'pred_caption.mp3'
    display.display(display.Audio(sound_file))

    # calculate BLEU score
    #print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# evaluate the skill of the model
def evaluate_model_basic_test_images(test_d,encoder_final_quant,encoder_interpreter,decoder_interpreter):
  print('evaluate_model_basic_test_images')
  actual, predicted = list(), list()
  cnt = 0
  start = time.time()
  for key, desc_list in test_d.items():
    cnt= cnt+1
    print(cnt)
    if encoder_final_quant == 1:
      print("\n ****** Encoder Only Quantization: ") 
      yhat = predict_captions_interpreter_encoder(key,encoder_interpreter)
    elif encoder_final_quant == 2:
      print("\n ****** NO Encoder Quantization and Final Model Quantization: ")
      yhat = predict_captions_interpreter_encoder_full(key,encoder_interpreter,decoder_interpreter,False)
    elif encoder_final_quant == 3:
      print("\n ****** Encoder and Final Model Quantization: ")
      yhat = predict_captions_interpreter_encoder_full(key,encoder_interpreter,decoder_interpreter,True)
    else:
      print("\n ****** Model Quantization: ")
      yhat = predict_captions(key)

    references = [desp.split() for desp in desc_list]
    actual.append(references)
    predicted.append(yhat.split())
    
    if cnt == 30 : 
        break
     #   print('ACTUAL- TEXT:')
      #  print(actual)
    
       # print('PREDICTED- TEXT:')
       # print(predicted)
        
  #print('FULL ACTUAL- TEXT: ')
  #print('FULL PREDICTED- TEXT: ')
  print("\n ****** Time taken: ", (time.time()-start)/60)  
  #calculate BLEU score
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def sample_image_test(interpreter_quant_encoder,interpreter_final):
     
    #evaluate_model_basic_oneimage('/home/nag/Nagaraj/Master/Flickr8k_Dataset/3320032226_63390d74a6.jpg','3320032226_63390d74a6.jpg',0,False,False)

    #evaluate_model_basic_oneimage('/home/nag/Nagaraj/Master/Flickr8k_Dataset/3320032226_63390d74a6.jpg','3320032226_63390d74a6.jpg',1,interpreter_quant_encoder,False)

    evaluate_model_basic_oneimage('/home/nag/Nagaraj/Master/Flickr8k_Dataset/3316725440_9ccd9b5417.jpg','3316725440_9ccd9b5417.jpg',2,interpreter_quant_encoder,interpreter_final)

    #evaluate_model_basic_oneimage('/home/nag/Nagaraj/Master/Flickr8k_Dataset/3320032226_63390d74a6.jpg','3320032226_63390d74a6.jpg',3,interpreter_quant_encoder,interpreter_final)


def Unseen_image_test(interpreter_quant_encoder,interpreter_final):
    #IMAGE_PATH = "/home/pi/Nagaraj/FaceRecognitionFinal/BhargavaRao_photo.jpg"
    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/NandiniPhoto.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)
  
    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/NagarajPhoto.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)

    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/KruthiPhoto.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)


    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/1.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)
    
    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/2.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)
    
    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/3.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)
    
    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/4.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)
    
    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/5.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)
    
    IMAGE_PATH = '/home/nag/Nagaraj/Master/test_images/6.jpg'
    
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,0,False,False)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,1,interpreter_quant_encoder,False)
    #evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,2,interpreter_quant_encoder,interpreter_final)
    evaluate_model_basic_oneimage_unseen_image(IMAGE_PATH,3,interpreter_quant_encoder,interpreter_final)
    

def beam_search_predictions_encoder_interpreter(image,encoder_interpreter, beam_index = 3):

    print('beam_search_predictions_encoder_interpreter')
    actual, predicted = list(), list()
    start_time = time.time()

    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            output = extract_features_quantized(encoder_interpreter,image)      
            preds = final_model.predict([output, np.array(par_caps)])
            #e = encoding_test[image[len(images):]]
            #preds = final_model.predict([np.array([e]), np.array(par_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

 
    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    #references = [desp.split() for desp in d[img_str_key]]
    #actual.append(references)
    #print(intermediate_caption)
    #predicted = final_caption
    #print('ACTUAL- TEXT:')
    #print(actual)
    #predicted_text = ' '.join(predicted[1:-1]

    final_caption = ' '.join(final_caption[1:])
    #predicted.append(final_caption.split())
    
    #print(len(predicted))
    #print(len(actual))
    
    #print('PREDICTED- TEXT:')
    #print(predicted)
    
    #print('FINAL- CAPTION:')
    #print(final_caption)
    
    #print("\n****** Time taken: ", (time.time()-start_time)/60)
      #calculate BLEU score
    #print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    
    return final_caption


def beam_search_predictions(image,beam_index = 3):
    
    print('beam_search_predictions')
    
    actual, predicted = list(), list()
    start_time = time.time()
        
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encoding_test[image[len(images):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    
    
    #references = [desp.split() for desp in d[img_str_key]]
    #references = [desp.split() for desp in desc_list]
    #actual.append(references)
    
    
    #print(intermediate_caption)
    #predicted = final_caption
    #print('ACTUAL- TEXT:')
    #print(actual)
    
    predicted.append(final_caption.split())
    
    print(len(predicted))
    #print(len(actual))
    
    print('PREDICTED- TEXT:')
    print(predicted)
    
    print('FINAL- CAPTION:')
    print(final_caption)
    
    print("\n****** Time taken: ", (time.time()-start_time)/60)
      #calculate BLEU score
    #print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))  
    
    return final_caption


def beam_search_predictions_encoder_interpreter_full(image,encoder_interpreter,decoder_interpreter,beam_index,yes_encoder):
  start = [word2idx["<start>"]]
  start_word = [[start, 0.0]]
    
  while len(start_word[0][0]) < max_len:
      temp = []
      for s in start_word:
          par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
          
          if yes_encoder == True:
            output = extract_features_quantized(encoder_interpreter,image)   
            preds = extract_final_model_quantized(decoder_interpreter,output,np.array(par_caps))
          else:
           output = encoding_test[image[len(images):]] 
           preds = extract_final_model_quantized(decoder_interpreter,np.array([output]),np.array(par_caps))

          #preds = final_model.predict([output, np.array(par_caps)])
          #e = encoding_test[image[len(images):]]
          #preds = final_model.predict([np.array([e]), np.array(par_caps)])
          word_preds = np.argsort(preds[0])[-beam_index:]
          
          # Getting the top <beam_index>(n) predictions and creating a 
          # new list so as to put them via the model again
          for w in word_preds:
              next_cap, prob = s[0][:], s[1]
              next_cap.append(w)
              prob += preds[0][w]
              temp.append([next_cap, prob])
                  
      start_word = temp
      # Sorting according to the probabilities
      start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
      # Getting the top words
      start_word = start_word[-beam_index:]
    
  start_word = start_word[-1][0]
  intermediate_caption = [idx2word[i] for i in start_word]

  final_caption = []
    
  for i in intermediate_caption:
      if i != '<end>':
          final_caption.append(i)
      else:
          break
  
  final_caption = ' '.join(final_caption[1:])
  return final_caption


def evaluate_model_basic_trained_bleu_images(encoder_interpreter, decoder_interpreter,encoder_final_quant,beam_index = 3):
  actual, predicted = list(), list()
  cnt = 0
  start = time.time()
  for key, desc_list in test_d.items():
    cnt= cnt+1
    print(cnt)
    if encoder_final_quant == 1:
      print("\n ****** Beam Encoder Only Quantization: ") 
      #yhat = predict_captions_interpreter_encoder(key,encoder_interpreter)
      yhat = beam_search_predictions_encoder_interpreter(key,encoder_interpreter,beam_index)
    elif encoder_final_quant == 2:
      print("\n ****** Beam NO Encoder Quantization and Final Model Quantization: ")
      yhat = beam_search_predictions_encoder_interpreter_full(key,encoder_interpreter,decoder_interpreter,beam_index,False)
    elif encoder_final_quant == 3:
      print("\n ****** Beam Encoder and Final Model Quantization: ")
      yhat = beam_search_predictions_encoder_interpreter_full(key,encoder_interpreter,decoder_interpreter,beam_index,True)
    else:
      print("\n ****** Beam Model Quantization: ")
      yhat = beam_search_predictions(key,beam_index)

    references = [desp.split() for desp in desc_list]
    actual.append(references)
    predicted.append(yhat.split())
    
    if cnt == 5 : 
        break
     #   print('ACTUAL- TEXT:')
      #  print(actual)
    
       # print('PREDICTED- TEXT:')
       # print(predicted)
        
  #print('FULL ACTUAL- TEXT: ')
  #print('FULL PREDICTED- TEXT: ')
  print("\n ****** Time taken: ", (time.time()-start)/60)  
  #calculate BLEU score
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))



if __name__ == '__main__':

    #camera = cv2.VideoCapture(0) # If you are using an USB Camera then Change use 1 instead of 0.
    #emotionVideo(camera)

    #IMAGE_PATH = "/home/pi/Nagaraj/FaceRecognitionFinal/BhargavaRao_photo.jpg"
    #IMAGE_PATH = "/home/pi/Nagaraj/FaceRecognitionFinal/Kavya1.jpg"
    
    #emotionImage(IMAGE_PATH) # If you are using this on an image please provide the path
    

    captions = open(token, 'r').read().strip().split('\n')

    d = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0])-2]
        if row[0] in d:
            d[row[0]].append(row[1])
        else:
            d[row[0]] = [row[1]]

    print(d['1000268201_693b08cb0e.jpg'])

    #tflite_models_dir = pathlib.Path("/content/drive/MyDrive/MasterCode")

    tflite_models_dir = pathlib.Path("/home/nag/Nagaraj/Master")
    
    #tflite_model_quant_file = tflite_models_dir/"batch32_Q_model.tflite"
    tflite_model_quant_file = tflite_models_dir/"Q_model.tflite"

    interpreter_quant_encoder = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter_quant_encoder.allocate_tensors()

    output_details = interpreter_quant_encoder.get_output_details()
    print('*********************\n')
    print(output_details)

    input_details = interpreter_quant_encoder.get_input_details()
    print('*********************\n')
    print(tf.__version__)
    print('*********************\n')
    print(input_details)

    #tflite_models_dir = pathlib.Path("/content/drive/MyDrive/MasterCode")
    tflite_models_dir = pathlib.Path("/home/nag/Nagaraj/Master")
    
    #tflite_model_quant_file = tflite_models_dir/"batch32_Q_model_full.tflite"
    tflite_model_quant_file = tflite_models_dir/"Q_model_full.tflite"
      
    interpreter_final = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter_final.allocate_tensors()
    output_details_final = interpreter_final.get_output_details()

    print(output_details_final)

    input_details_final = interpreter_final.get_input_details()

    print(input_details_final)

    #unique = pickle.load(open('/content/drive/MyDrive/MasterCode/unique.p', 'rb'))
    unique = pickle.load(open('/home/nag/Nagaraj/Master/unique.p', 'rb'))  

    print(len(unique))

    word2idx = {val:index for index, val in enumerate(unique)}

    print(word2idx['<start>'])

    idx2word = {index:val for index, val in enumerate(unique)}

    print(idx2word[5553])

    #max_len = 0
    #for c in caps:
    #    c = c.split()
    #    if len(c) > max_len:
    #        max_len = len(c)
    #max_len

    max_len = 40

    #images = '/content/drive/MyDrive/MasterCode/Flickr8k_Dataset/'
    #/home/pi/Nagaraj/MasterProgram
    images = '/home/nag/Nagaraj/Master/Flickr8k_Dataset/'

    # Contains all the images
    img = glob.glob(images+'*.jpg')

    print(img[1])

    test_images_file = '/home/nag/Nagaraj/Master/Flickr8k_text/Flickr_8k.testImages.txt'
    test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

    # Getting the testing images from all the images
    test_img = split_data(test_images)
    print(len(test_img))

    encoding_test = pickle.load(open('/home/nag/Nagaraj/Master/encoded_images_test_inceptionV3.p', 'rb'))

    encoding_test[test_img[0][len(images):]].shape
    
    test_d = {}
    for i in test_img:
        if i[len(images):] in d:
            test_d[i] = d[i[len(images):]]
            
    
    #SAVE_MODEL_FOLDERPATH = '/home/nag/Nagaraj/MasterProgram/my_modelBase-InceptionV3-baseline-LSTM30'

    #final_model = tf.keras.models.load_model('/home/nag/Nagaraj/MasterProgram/my_modelBase-InceptionV3-baseline-LSTM30/')
    
    final_model = tf.keras.models.load_model('/home/nag/Nagaraj/Master/H5_modelBase-InceptionV3-baseline-LSTM_ep_30.h5')

    #final_model = tf.keras.models.load_model('/home/nag/Nagaraj/MasterProgram/H5_model_batch32Batch32_Base-InceptionV3-baseline-LSTM_ep_5.h5')
    
    # Check its architecture
    #print(final_model.summary())


    #im = '/home/nag/Nagaraj/Master/Flickr8k_Dataset/2542662402_d781dd7f7c.jpg'
    #print ('Normal Max search:', predict_captions(im))
    #print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
    #print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
    #print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
    #Image.open(im)


    #testfeature_quant = extract_features_quantized(interpreter_quant_encoder,im)
    #print(testfeature_quant)


    #evaluate_model_basic_test_images(test_d,0,interpreter_quant_encoder,interpreter_final)
    #evaluate_model_basic_test_images(test_d,1,interpreter_quant_encoder,interpreter_final)
    #evaluate_model_basic_test_images(test_d,2,interpreter_quant_encoder,interpreter_final)
    #evaluate_model_basic_test_images(test_d,3,interpreter_quant_encoder,interpreter_final)
    
    sample_image_test(interpreter_quant_encoder,interpreter_final)
    
    
    #Unseen_image_test(interpreter_quant_encoder,interpreter_final)
    
        
    #im = '/home/nag/Nagaraj/Master/Flickr8k_Dataset/2542662402_d781dd7f7c.jpg'
    #print ('Normal Max search:', predict_captions_interpreter_encoder(im,interpreter_quant_encoder))
    #Image.open(im)
    #print ('Normal Max search:', predict_captions(im))    
    #print('Without Optimzation Beam Search, k=3:', beam_search_predictions(im,'2542662402_d781dd7f7c.jpg',beam_index=3))
    #print('Without Optimzation Beam Search, k=5:', beam_search_predictions(im,'2542662402_d781dd7f7c.jpg',beam_index=5))
    #print('Without Optimzation Beam Search, k=7:', beam_search_predictions(im,'2542662402_d781dd7f7c.jpg',beam_index=7))
    #print('Beam Search, k=3:', beam_search_predictions_encoder_interpreter(im,'2542662402_d781dd7f7c.jpg',interpreter_quant_encoder, beam_index=3))
    #print('Beam Search, k=5:', beam_search_predictions_encoder_interpreter(im,'2542662402_d781dd7f7c.jpg',interpreter_quant_encoder, beam_index=5))
    #print('Beam Search, k=7:', beam_search_predictions_encoder_interpreter(im,'2542662402_d781dd7f7c.jpg',interpreter_quant_encoder, beam_index=7))

    #evaluate_model_basic_trained_bleu_images(interpreter_quant_encoder,interpreter_final,0,beam_index = 3)
    #evaluate_model_basic_trained_bleu_images(interpreter_quant_encoder,interpreter_final,1,beam_index = 3)
    #evaluate_model_basic_trained_bleu_images(interpreter_quant_encoder,interpreter_final,2,beam_index = 3)
    #evaluate_model_basic_trained_bleu_images(interpreter_quant_encoder,interpreter_final,3,beam_index = 3) 