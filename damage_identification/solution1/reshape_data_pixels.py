'''
Created on May 31, 2016

@author: pankajrawat
'''
import pandas as pd
import cv2
import numpy as np
#print df.head(5)


def display (img, wait=True, label='dst_rt', time=None):
    cv2.imshow(label, img)
    if not any([wait, time]):
        return
    if not wait and time:
        cv2.waitKey(time)
        return
    cv2.waitKey(0)


def reshapeDataFile(originalFile, outputFile, newDimentionPixels):
    df = pd.read_csv(originalFile)
    skip = 9
    cols = ['defect', 'width', 'height', 'Cx', 'Cy', 'area', 'isConvex', 'extent', 'solidity']
    pixelsCols = map(lambda x: 'px_' + str(x) , range(newDimentionPixels ** 2))
    cols.extend(pixelsCols)
    
    dfM = None
    for index, row in df.iterrows():
        #print row[0:9]
        img = np.array([row[9:]])
        img = img.reshape(64, 64)
        #display(img)
        resized = cv2.resize(img, (newDimentionPixels, newDimentionPixels), interpolation = cv2.INTER_AREA)
        #display(resized)
        resized = resized.flatten()
        data = np.concatenate([row[0:9], resized], axis=0)
        dfT = pd.DataFrame(data=[data], columns=cols)
        dfM = dfT if dfM is None else pd.concat([dfM, dfT], axis=0)
        print dfM.shape

    #dfM.to_csv(outputFile, index=False, mode='a', header=True)
    dfM.to_csv(outputFile, index=False)

reshapeDataFile('dataset/data.csv', 'dataset/data_32p.csv', 32)

