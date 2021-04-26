#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the required libraries
import numpy as np
import pickle
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model


# In[2]:


# To avoid error due to tensorflow-gpu backend
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[3]:


# Resnet model pretrained on imagenet
model_res=ResNet50(weights="imagenet", input_shape=(224, 224, 3))
model_new=Model(model_res.input, model_res.layers[-2].output)


# In[4]:


# Image preprocessing
def preprocess_img(img):
    img=image.load_img(img, target_size=(224, 224))
    img=image.img_to_array(img)
    img=np.expand_dims(img, axis=0)
    img=preprocess_input(img)
    return img


# In[5]:


# Encoding images
def encode_image(img):
    img=preprocess_img(img)
    feature_vector=model_new.predict(img)
    feature_vector=feature_vector.reshape((1, -1)) # Reshaping to (1, 2048)
    return feature_vector


# In[6]:


model=load_model("model_final_weights.h5")


# In[7]:


with open("vocab/w2i.pkl", 'rb') as f:
    word_to_idx=pickle.load(f)
with open("vocab/i2w.pkl", 'rb') as f:
    idx_to_word=pickle.load(f)


# In[8]:


max_len=35


# In[9]:


def predict_caption(photo):
    inp_text="<s>"
    for i in range(max_len):
        sequence=[word_to_idx[w] for w in inp_text.split() if w in word_to_idx]
        sequence=pad_sequences([sequence], maxlen=max_len, padding='post')
        ypred=model.predict([photo, sequence])
        ypred=ypred.argmax()
        word=idx_to_word[ypred]
        inp_text+=(" "+word)
        if word =='<e>':
            break
    final_caption=inp_text.split()[1: -1]
    final_caption=" ".join(final_caption)
    final_caption=final_caption.capitalize()+"."
    return final_caption


# In[10]:


def caption_image(image):
    enc=encode_image(image)
    caption=predict_caption(enc)
    return caption


# In[ ]:


if __name__=="__main__":
    print(caption_image("sample_image.jpg"))


# In[ ]:




