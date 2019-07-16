import sys
import shutil
import numpy as np
import os
import cv2
from pdf2jpg import pdf2jpg
from multiprocessing.pool import Pool

def convert(x):
    try:
        sourcefile, outputdir = x
        #shutil.rmtree(os.path.join(outputdir, os.path.basename(sourcefile)))
        dirtocheck = os.path.join(outputdir, os.path.basename(sourcefile))
        if os.path.exists(dirtocheck):
           print("Skipping", sourcefile)
           return 
        print("Processing", sourcefile)

        result = pdf2jpg.convert_pdf2jpg(sourcefile, outputdir, dpi="80", pages="0,1,2,3")
        output_jpgs = result[0]["output_jpgfiles"]
        imgs = [cv2.imread(x) for x in output_jpgs]
        #print([x.shape for x in imgs])
        imgs = [cv2.resize(x, (876, 683), interpolation = cv2.INTER_AREA) for x in imgs]
        diff = 4 - len(output_jpgs) 
        blank_image_buffer = [np.zeros((683, 876, 3), np.uint8)] * diff

        imgs.extend(blank_image_buffer)

        img = np.vstack(imgs)
        outputfilename = os.path.join(outputdir, os.path.basename(sourcefile), "stacked_image.jpg")
        cv2.imwrite(outputfilename, img)
        for output_jpg in output_jpgs:
            os.remove(output_jpg)
        #print(img.shape)
    except Exception as err:
        print(err) 

def main():
    srcdir = sys.argv[1]
    outputdir = sys.argv[2]
    samples_start = int(sys.argv[3])
    samples_end = int(sys.argv[4])
    outputdir = os.path.join(outputdir, os.path.basename(srcdir))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    print("SRC => ", srcdir)
    print("DST => ", outputdir)
    print("Samples => ", samples_start, samples_end)

    sourcefiles = list(map(lambda x: os.path.join(srcdir, x) , os.listdir(srcdir)))
    sourcefiles = sorted(sourcefiles)
    print("Total sourcefiles =>", len(sourcefiles))
    sourcefiles = sourcefiles[samples_start:samples_end]
    print("Files already presebt =>", len(os.listdir(outputdir)))
    sourcefiles = zip(sourcefiles, [outputdir] * len(sourcefiles))
    p = Pool(5)
    p.map(convert, sourcefiles)
    p.terminate()

if  __name__  == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
