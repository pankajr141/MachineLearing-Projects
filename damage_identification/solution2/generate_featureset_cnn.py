'''
Created on Jun 30, 2016

@author: pankajrawat
'''

import os
import cv2
import numpy as np
import pandas as pd
import damage.extract_damage as ed
from damage.utils import common
from image import operations as imop


IMAGE_DIMENTION = 64
def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]


def getDamageDF(path, defect):    
    imageData = None
    for fileName in os.listdir(path):
        #print fileName
        filePath = os.path.join(path, fileName)
        #print filePath
        image = cv2.imread(filePath)
        image = common.getResizedImageScaled(image, IMAGE_DIMENTION)
        orig = image.copy()
        image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
        
        #print filePath, features
        #imop.display(image)
         
        if len(image.shape) == 2:
            zLen = 1
        else:
            zLen = image.shape[2]
        
        image = image.reshape(1, image.size)

        if imageData is None:
            imageData = image
            continue
        imageData = np.concatenate((imageData, image), axis=0)
    cols = map(lambda x: 'px_' + str(x) , range((IMAGE_DIMENTION ** 2) * zLen))
    #cols.extend(['avgBlackArea'])
    print "Samples >> ", imageData.shape, " ||| ", path
    df = pd.DataFrame(imageData, columns=cols)
    df['defect'] = defect
    df = change_column_order(df, 'defect', 0)
    return df


def main():
    outputFile = os.path.join("D:\\New Volume\\workspace\\damage\\dataset", "data_64_cnn.csv")
    if os.path.exists(outputFile):
        os.remove(outputFile)
    
    header = True if not os.path.exists(outputFile) else False
    imagesDir = "D:\\New Volume\\Documents\\Damages"
    damagePaths = []
    nonDamagePaths = []
    for path in os.listdir(imagesDir):
        if "sysdamages_" in path or "nondamages" in path:
            path = os.path.join(imagesDir, path)
            nonDamagePaths.append(path)
            continue
        elif "Mdamages" in path:
            path = os.path.join(imagesDir, path)
            damagePaths.append(path)
    
    print damagePaths
    print nonDamagePaths

    damagePaths =    [
                        'D:\\New Volume\\Documents\\Damages\\Mdamages', 
                        'D:\\New Volume\\Documents\\Damages\\Mdamages_2', 
                        'D:\\New Volume\\Documents\\Damages\\Mdamages_3', 
                        'D:\\New Volume\\Documents\\Damages\\Mdamages_4', 
                      ]

    nonDamagePaths = [
                      #'D:\\New Volume\\Documents\\Damages\\nondamages', 
                      'D:\\New Volume\\Documents\\Damages\\sysdamages_1',
                      'D:\\New Volume\\Documents\\Damages\\sysdamages_2',
                      'D:\\New Volume\\Documents\\Damages\\sysdamages_3',
                      'D:\\New Volume\\Documents\\Damages\\sysdamages_4',
                      'D:\\New Volume\\Documents\\Damages\\sysdamages_5',
                      'D:\\New Volume\\Documents\\Damages\\sysdamages_6',
                      ]

    #damagePaths = filter(lambda x: not 'C' in x, damagePaths)
    df1 = None
    for damagePath in damagePaths:
        dfT = getDamageDF(damagePath, 1)
        if df1 is None:
            df1 = dfT
            continue
        df1 = pd.concat([df1, dfT], axis=0)
    #df1 = pd.concat([df1,  df1], axis=0)
    df2 = None
    for nondamagePath in nonDamagePaths:
        dfT = getDamageDF(nondamagePath, 0)
        if df2 is None:
            df2 = dfT
            continue
        df2 = pd.concat([df2, dfT], axis=0)
    
    df1Sampled = df1   
    df2Sampled = df2

    if len(df2) >  len(df1): 
        allowedNegativeSamples = int(len(df1) * 1.5)
        if len(df2) > allowedNegativeSamples:
            df2Sampled = df2.sample(n=allowedNegativeSamples)
    else:
        allowedPostiveSamples = len(df2) * 2
        if len(df1) > allowedPostiveSamples:
            df1Sampled = df1.sample(n=allowedPostiveSamples)
    df = pd.concat([df1Sampled, df2Sampled], axis=0)
    
    print "Total Positve    >>", df1.shape
    print "Total Negative   >>", df2.shape
    print "Samples Postive  >>", df1.shape
    print "Samples Negative >>", df2Sampled.shape
    print "Samples Overall  >>", df.shape
    df.to_csv(outputFile, index=False, mode='a', header=header)

if __name__ == "__main__":
    main()