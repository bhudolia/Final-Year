

import scrapy
filename = "review.txt"
import urllib
import json
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
             return(json_data['imdbID'])
    except urllib.error.URLError as e:
        print(f"ERROR: {e.reason}")


#with open('APIkeys.json') as f:
#keys = json.load(f)
omdbapi = 'a7b69699'
serviceurl = 'http://www.omdbapi.com/?'
apikey = '&apikey='+omdbapi        
imdbid = search_movie('Black Panther')

url = 'https://www.imdb.com/title/'+imdbid+'/reviews'

class ImdbSpider(scrapy.Spider):
    name = "imdb_spider"
    
    def start_requests(self):
        
        yield scrapy.Request(url=url, callback=self.parse)
            
    def parse(self, response):
        
        book_list = response.xpath('//*[@id="main"]/section/div/div/div/div/div/div/div[1]/text()').extract()
        
        with open(filename, 'a+') as f:
            for book_title in book_list:
                if book_title: f.write(book_title)
					
					
                