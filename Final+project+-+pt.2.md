

```python
import numpy
import pandas as pd
#import theano
#import keras 
```


```python
df = pd.read_pickle(r'data')
df.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple iPhone 6 Plus 64GB Space Gray LTE Cellul...</td>
      <td>Cell Phones &amp; Smartphones</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apple iPhone 5S 16GB Space Gray LTE Cellular A...</td>
      <td>Cell Phones &amp; Smartphones</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple iPhone 6 64GB Space Gray LTE Cellular AT...</td>
      <td>Cell Phones &amp; Smartphones</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apple iPhone 6 64GB Silver LTE Cellular AT&amp;T M...</td>
      <td>Cell Phones &amp; Smartphones</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apple iPhone 5S 16GB Silver LTE Cellular Sprin...</td>
      <td>Cell Phones &amp; Smartphones</td>
    </tr>
  </tbody>
</table>
</div>




```python
category = [[],[],[],[]]
for row in df.iterrows():
    cat = row[1].Category
    title = row[1].Title
    if cat == "Cell Phones & Smartphones":
        category[0].append(title)
    elif cat == "Cookbooks":
        category[1].append(title)
    elif cat == "Records":
        category[2].append(title)
    else: #Video Games
        category[3].append(title)
```


```python
print len(category[0])
print len(category[1])
print len(category[2])
print len(category[3])
```

    561
    422
    2120
    228
    


```python
list_to_string = []
for i in range (0,4):
    list_to_string.append(" ~ ".join(category[i]))
```

keras has a problem with text that isn't unicode, so we found a patch for the problem:


```python
#from keras.preprocessing.text import text_to_word_sequence
import keras.preprocessing.text

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
    else:
        translate_table = maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]
    
keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence
```

    Using Theano backend.
    


```python
words = [] 
for i in range (0,4):
    words.append(text_to_word_sequence(list_to_string[i], lower=False, split=" "))
```


```python
from keras.preprocessing.text import Tokenizer
text_mtx = []
token = Tokenizer(nb_words=500,char_level=False)
for i in range (0,4):
    token.fit_on_texts(words[i])
    text_mtx.append(token.texts_to_matrix(words[i], mode='binary'))
    #print text_mtx[i].shape
```

    C:\Users\matoo\Anaconda2\lib\site-packages\keras\preprocessing\text.py:139: UserWarning: The `nb_words` argument in `Tokenizer` has been renamed `num_words`.
      warnings.warn('The `nb_words` argument in `Tokenizer` '
    


```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
```

for the first category:


```python
text = text_mtx[0]
input_ = text
output_ = text

model1 = Sequential()
model1.add(Embedding(input_dim=input_.shape[1],output_dim= 42, input_length=input_.shape[1]))
model1.add(Flatten())
model1.add(Dense(output_.shape[1], activation='sigmoid'))
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
model1.fit(input_, y=output_, batch_size=300, nb_epoch=10, verbose=1, validation_split=0.2)
```

    C:\Users\matoo\Anaconda2\lib\site-packages\keras\models.py:844: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      warnings.warn('The `nb_epoch` argument in `fit` '
    

    Train on 3656 samples, validate on 915 samples
    Epoch 1/10
    3656/3656 [==============================] - 7s - loss: 5.2635 - acc: 0.0197 - val_loss: 6.5042 - val_acc: 0.0066
    Epoch 2/10
    3656/3656 [==============================] - 7s - loss: 4.9931 - acc: 0.0260 - val_loss: 6.6390 - val_acc: 0.0098
    Epoch 3/10
    3656/3656 [==============================] - 8s - loss: 4.9402 - acc: 0.0350 - val_loss: 6.7318 - val_acc: 0.0066
    Epoch 4/10
    3656/3656 [==============================] - 6s - loss: 4.8927 - acc: 0.0260 - val_loss: 6.7665 - val_acc: 0.0066
    Epoch 5/10
    3656/3656 [==============================] - 6s - loss: 4.8367 - acc: 0.0293 - val_loss: 7.1025 - val_acc: 0.0066
    Epoch 6/10
    3656/3656 [==============================] - 6s - loss: 4.7681 - acc: 0.0416 - val_loss: 6.7622 - val_acc: 0.0109
    Epoch 7/10
    3656/3656 [==============================] - 6s - loss: 4.6563 - acc: 0.0818 - val_loss: 7.5684 - val_acc: 0.0230
    Epoch 8/10
    3656/3656 [==============================] - 6s - loss: 4.5108 - acc: 0.2019 - val_loss: 7.6558 - val_acc: 0.0448
    Epoch 9/10
    3656/3656 [==============================] - 6s - loss: 4.3601 - acc: 0.2883 - val_loss: 7.5444 - val_acc: 0.0831
    Epoch 10/10
    3656/3656 [==============================] - 6s - loss: 4.2006 - acc: 0.3515 - val_loss: 7.7496 - val_acc: 0.1180
    




    <keras.callbacks.History at 0x12ec9b38>




```python
def get_next(text,token,model,fullmtx,fullText):
    tmp = text_to_word_sequence(text, lower=False, split=" ")
    tmp = token.texts_to_matrix(tmp, mode='binary')
    p = model.predict(tmp)
    bestMatch = np.min(np.argmax(p))
    next_idx = np.min(np.where(fullmtx[:,bestMatch]>0))
    return fullText[next_idx]
```


```python

```
