
Importing the packages we need


```python
import pandas as pd
import ebaysdk
from ebaysdk.finding import Connection as finding
import pickle
```

## Part a: collecting the data

We use ebay sdk package in order to collect all the records from three categories - women boots(53557), Video Games (139973), Cell Phones & Smartphones (9355). From each page we can take max 100 items so we loop over 20 pages, that gives us a total of max 2000 records.


```python
api = finding(domain='svcs.sandbox.ebay.com', appid='MaayanMa-DSprojec-SBX-e8e06383e-67abf65d', config_file=None)
item_id, title, cat_id=[], [], []
#category ID, name - 53557,women boots; 139973,Video Games; 9355, Cell Phones & Smartphones
try:
    for pn in range (1,20):
        api.execute('findItemsAdvanced', {'categoryId' : ['53557','9355','139973'], 
                                          'paginationInput': {'entriesPerPage': '100', 'pageNumber': pn}})
        dictstr = api.response_dict()
        for item in dictstr.searchResult.item:
            item_id.append(item.itemId)
            title.append(item.title)
            cat_id.append(item.primaryCategory.categoryId)
except AttributeError:
    print "total pages available: ", pn
```

Inserting all the records into a Data Frame


```python
df = pd.DataFrame()
df['ItemID'] = item_id
df['Title'] = title
df['CategoryID'] = cat_id
```

Looking at our databaes


```python
print "Total records: ", df.shape[0], "\n"
print "Records from each category:"
print df.groupby('CategoryID').count()
```

    Total records:  1900 
    
    Records from each category:
                ItemID  Title
    CategoryID               
    139973         408    408
    53557         1182   1182
    9355           310    310
    

Saving our database in a pickle file for later use


```python
df.to_pickle(r'data_test')
```

## Part b: building a classifier


```python

```
