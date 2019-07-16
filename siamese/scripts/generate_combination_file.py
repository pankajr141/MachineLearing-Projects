import os
import sys
import random

def getAllFilPathsFromDirectory(directory):
    """ Function to list all the files present inside folder and nested directories"""
    
    filePathinPool = []        
    for cntr, file_ in enumerate(os.listdir(directory)):
        if "attach" in file_:
            continue
        filePath = os.path.join(directory, file_)
        if not os.path.isfile(filePath):
            subDirFilePaths = getAllFilPathsFromDirectory(filePath)
            if subDirFilePaths:
                filePathinPool.extend(subDirFilePaths)
            continue
        filePathinPool.append(filePath)
    #filePathinPool = filter(lambda x: not "attach" in x, filePathinPool)
    return filePathinPool

def main():
    srcdir = sys.argv[1]
    sourcetypes = list(map(lambda x:  os.path.join(srcdir, x),os.listdir(srcdir)))
    mapping = []
    for sourcetype in sourcetypes:
        sourcetypefiles = getAllFilPathsFromDirectory(sourcetype)
        mapping.append(sourcetypefiles)


    numrecords = int(sys.argv[2])
    outputfile = sys.argv[3]
    print("Number of records to generate =>", numrecords)

    fp = open(outputfile, 'w')
    cntr = 0
    while cntr < numrecords:
        try:
            similar_index = random.randint(0, len(mapping)-1)
            different_index = random.randint(0, len(mapping)-1)
            while similar_index == different_index:
                print("REDOING => ", similar_index, different_index)
                different_index = random.randint(0, len(mapping))
            #print(similar_index, different_index, len(mapping[similar_index]), len(mapping[different_index])) 
            #print(set(mapping[similar_index]))
            similars = random.sample(set(mapping[similar_index]), 2)
            different = random.sample(set(mapping[different_index]), 1)
            #print("%s,%s,%s\n" % (similars[0], similars[1], different[0]))
            fp.write("%s,%s,%s\n" % (similars[0], similars[1], different[0])) 
            cntr += 1
        except Exception as err:
            print(cntr, err)

    fp.close()
if  __name__  == "__main__":
    main()
