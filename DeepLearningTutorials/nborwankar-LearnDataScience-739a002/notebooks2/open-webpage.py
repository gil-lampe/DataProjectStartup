# open-webpage.py

import urllib2

url = 'http://www.oldbaileyonline.org/browse.jsp?id=t17800628-33&div=t17800628-33'

response = urllib2.urlopen(url)
webContent = response.read()

print(webContent[0:300])