from bs4 import BeautifulSoup
import requests

url = 'https://timesofindia.indiatimes.com/india/timestopten.cms'

result = requests.get(url)
soup = BeautifulSoup(result.content, 'lxml')

content = soup.find(class_='outertable')

l = content.findAll(class_='news_title')       # List
series = [ele.get_text() for ele in l]
print(series)