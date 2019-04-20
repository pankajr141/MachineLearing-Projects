'''
Created on Jul 8, 2016

@author: pankajrawat
'''

''


import requests
url = 'http://10.175.13.222:8080/datawing/saveImage'
data = '{"imageId":"12225","batchId":"23224","roofId":"2222","imagePath":"abc","uploadStatus":"success","errorId":"1233","errorMessage":"error","imageName":"roof3.jpg","coordinates":["1","2","3"]}'
response = requests.post(url, data=data)
print response
