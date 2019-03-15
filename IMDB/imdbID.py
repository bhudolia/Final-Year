#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[15]:


import urllib
import json


# In[18]:



def search_movie(title):
    if len(title) < 1 or title=='quit':
        print('Goodbye now…')
        return None
    try:
        url = serviceurl + urllib.parse.urlencode({'t': title})+apikey
        print(f'Retrieving the data of “{title}” now… ')
        uh = urllib.request.urlopen(url)
        data = uh.read()
        json_data=json.loads(data)

        if json_data['Response']=='True':
             print(json_data['imdbID'])
    except urllib.error.URLError as e:
        print(f"ERROR: {e.reason}")


#with open('APIkeys.json') as f:
#keys = json.load(f)
omdbapi = 'a7b69699'
serviceurl = 'http://www.omdbapi.com/?'
apikey = '&apikey='+omdbapi        
search_movie('room')


# In[ ]:




