# trial-content.py

import urllib2, obo

url = 'http://www.oldbaileyonline.org/browse.jsp?id=t17800628-33&div=t17800628-33'

response = urllib2.urlopen(url)
HTML = response.read()

print(obo.stripTags(HTML))