
# Final project

our goal is to find and classify item descriptions from eBay and generate a new description for every classification we have (three categories).

Importing the packages we need


```python
import numpy as np
import pandas as pd
import ebaysdk
from ebaysdk.finding import Connection as finding
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import nltk
from nltk.corpus import stopwords
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn

from sklearn.svm import SVC
import sklearn.naive_bayes as nb
from sklearn import neighbors 
from sklearn.tree import DecisionTreeClassifier
```

## Part a: collecting the data

We use ebay sdk package in order to collect all the records from three categories - Cookbooks (11104), Records (176985), Video Games (139973), Cell Phones & Smartphones (9355). From each page we can take max 100 items so we loop over 20 pages, that gives us a total of max 2000 records.


```python
def collect_data(cat):
    try:
        for pn in range (1,50):
            api.execute('findItemsAdvanced', {'categoryId' : cat, 
                                              'paginationInput': {'entriesPerPage': '100', 'pageNumber': pn}})
            dictstr = api.response_dict()
            for item in dictstr.searchResult.item:
                item_id.append(item.itemId)
                title.append(item.title)
                cat.append(item.primaryCategory.categoryName)
    except AttributeError:
        print "total pages available: ", pn
```

Inserting all the records into a Data Frame


```python
df = pd.DataFrame()
api = finding(domain='svcs.sandbox.ebay.com', appid='MaayanMa-DSprojec-SBX-e8e06383e-67abf65d', config_file=None)
item_id, title, cat = [], [], []
collect_data(['9355','139973'])
collect_data(['11104','176985'])
df['ItemID'] = item_id
df['Title'] = title
df['Category'] = cat
```

Looking at our databaes


```python
df.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ItemID</th>
      <th>Title</th>
      <th>CategoryID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>110205984389</td>
      <td>Apple iPhone 6 Plus 64GB Space Gray LTE Cellul...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>1</th>
      <td>110205978911</td>
      <td>Apple iPhone 5S 16GB Space Gray LTE Cellular A...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>110205982787</td>
      <td>Apple iPhone 6 64GB Space Gray LTE Cellular AT...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>3</th>
      <td>110205984011</td>
      <td>Apple iPhone 6 64GB Silver LTE Cellular AT&amp;T M...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>4</th>
      <td>110205985297</td>
      <td>Apple iPhone 5S 16GB Silver LTE Cellular Sprin...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>5</th>
      <td>110205983931</td>
      <td>Apple iPhone 5 16GB Black LTE Cellular Straigh...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>6</th>
      <td>110205982867</td>
      <td>Apple iPhone 6 16GB Gold LTE Cellular Sprint M...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>7</th>
      <td>110205985429</td>
      <td>Apple iPhone 6 128GB Gold LTE Cellular Verizon...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>8</th>
      <td>110198623979</td>
      <td>Apple iPhone 6 Plus 64GB Silver LTE Cellular S...</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>9</th>
      <td>110198624555</td>
      <td>Apple iPhone 6S 64GB Space Gray LTE Cellular A...</td>
      <td>9355</td>
    </tr>
  </tbody>
</table>
</div>



Drop duplicates if exists. We will find duplicates according to ItemID


```python
print "number of rows before: ", df.shape[0]
df.drop_duplicates(subset='ItemID', keep='first', inplace=True)
print "number of rows after: ", df.shape[0]
df = df.drop('ItemID',axis = 1)
```

    number of rows before:  3331
    number of rows after:  3331
    

Checking if there are NULLs that needs to be fixed


```python
df.isnull().sum()
```




    ItemID        0
    Title         0
    CategoryID    0
    dtype: int64



There are no NULL values in our data.

Let's look at our Categories - how many do we have from each one and what part do they take. 


```python
print df.groupby('CategoryID').count()
df['CategoryID'] = df['CategoryID'].astype('category')
colors = ['blue', 'green','red','turquoise']
labels = df.CategoryID.unique()
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
df['CategoryID'].value_counts().plot(kind='bar', color = colors)
ax2 = fig.add_subplot(122)
df['CategoryID'].value_counts().plot(kind = 'pie')
plt.show()
```

                Title
    CategoryID       
    11104         422
    139973        228
    176985       2120
    9355          561
    


![png](output_15_1.png)


Saving our database in a pickle file for later use


```python
df.to_pickle(r'data')
```

## Part b: building a classifier

first, we will load our database from the last part


```python
df = pd.read_pickle(r'data')
print df.head(5)
```

                                                   Title CategoryID
    0  Apple iPhone 6 Plus 64GB Space Gray LTE Cellul...       9355
    1  Apple iPhone 5S 16GB Space Gray LTE Cellular A...       9355
    2  Apple iPhone 6 64GB Space Gray LTE Cellular AT...       9355
    3  Apple iPhone 6 64GB Silver LTE Cellular AT&T M...       9355
    4  Apple iPhone 5S 16GB Silver LTE Cellular Sprin...       9355
    

This function plots a confusion matrix


```python
def plot_cnf_matrix(cms, classes, model_name, vec_size):
    df_cm = pd.DataFrame(cms, index = classes, columns = classes)
    plt.figure(figsize = (5,5))
    fig, ax = plt.subplots()
    ax = sn.heatmap(df_cm, annot=True, linewidths=.5, fmt='g')
    print model_name, " for ", vec_size, "features"
    plt.show()
```

This function returns the train and test sets


```python
def extract_test_set(df, numOfFeatures, RemoveStopWords):
    stops = set(stopwords.words('english'))
    corpus = np.array(df.Title.values).tolist()
    target = np.array(df['CategoryID'].values)
    if RemoveStopWords:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=numOfFeatures, analyzer='word', ngram_range=(1, 1))
    else:
        vectorizer = TfidfVectorizer(max_features=numOfFeatures, analyzer='word', ngram_range=(1, 1))
    
    X = vectorizer.fit_transform(corpus)
#    global feature_names
    feature_names = vectorizer.get_feature_names()
    return X, target, feature_names
```

This function classifies a given model on a given data and target


```python
def Classify(vector_length, model, data, target, kf, mod_name,plot_cn):
    cm = []
    error = []
    for train_indices, test_indices in kf:
        # Get the dataset; this is the way to access values in a pandas DataFrame
        train_X = data[train_indices, :]
        train_Y = target[train_indices]
        test_X = data[test_indices, :]
        test_Y = target[test_indices]
        # Train the model
        model.fit(train_X, train_Y)
        predictions = model.predict(test_X)
        # Evaluate the model
        classes = model.classes_                
        cm.append(confusion_matrix(test_Y, predictions, labels=classes))
        error.append(model.score(test_X, test_Y))
    accuracy = np.mean(error)
    if plot_cn == True:
        for i in range(0,9):
            cms = np.mean(cm, axis=0)
        plot_cnf_matrix(cms, classes, mod_name, vector_length)
    return accuracy
```

First, we will check four algorithms with different vector sizes in order to determin which combinations gives us a good result


```python
vector_length = [5, 7, 10, 20, 50, 100]

nbayes = nb.MultinomialNB()
dtree = DecisionTreeClassifier(random_state=0, max_depth=20)
svm = SVC(decision_function_shape='ovo',kernel='linear')
knn = neighbors.KNeighborsClassifier()

models = [nbayes, dtree, knn, svm]
mod_names = ['Naive Bayes', 'DT', 'KNN', 'Linear SVM']

nfolds = 10

a = {'Classifer': pd.Series(index=mod_names),}
scoreTable = pd.DataFrame(a)
scoreTable.__delitem__('Classifer')

for vl in vector_length:
    data, target, features = extract_test_set(df, vl, False)
    kf = KFold(data.shape[0], n_folds = nfolds, shuffle = True, random_state = 1)
    accuracies = []
    index = 0
    for mod in models:
        accuracies.append(Classify(vl, mod, data, target, kf, mod_names[index], False))
        index += 1
    scoreTable[vl] = accuracies

print scoreTable
```

                      5         7         10        20        50        100
    Naive Bayes  0.685983  0.680576  0.771844  0.874813  0.948361  0.958269
    DT           0.904230  0.909036  0.915939  0.948360  0.969075  0.968175
    KNN          0.901828  0.910237  0.913237  0.943856  0.964273  0.963974
    Linear SVM   0.904831  0.912337  0.916840  0.949862  0.969976  0.971177
    

We can see that a vector length of 20 gives us an accuracy that is over 80% with all four algorithms.
now, we'll look at some confusion matrices.

The first confusion matrix will be with Naive Bayes:


```python
accuracies.append(Classify(20, nbayes, data, target, kf, 'Naive Bayes', True))
```

    Naive Bayes  for  20 features
    


    <matplotlib.figure.Figure at 0xdd155f8>



![png](output_30_2.png)


The second confusion matrix will be with Linear SVM:


```python
accuracies.append(Classify(20, svm, data, target, kf, 'Linear SVM', True))
```

    Linear SVM  for  20 features
    


    <matplotlib.figure.Figure at 0xe43e6a0>



![png](output_32_2.png)


The third confusion matrix will be with deceision tree:


```python
accuracies.append(Classify(20, dtree, data, target, kf, 'DT', True))
```

    DT  for  20 features
    


    <matplotlib.figure.Figure at 0xe256940>



![png](output_34_2.png)


The last confusion matrix will be with KNN:


```python
accuracies.append(Classify(20, knn, data, target, kf, 'KNN', True))
```

    KNN  for  20 features
    


    <matplotlib.figure.Figure at 0xe0036d8>



![png](output_36_2.png)

