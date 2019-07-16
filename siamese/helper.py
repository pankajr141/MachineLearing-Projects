import os
from pdf2jpg import pdf2jpg
import cv2
import numpy as np

def convert_tojpg(sourcefile, outputdir="tmp"):
    try:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
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
        return outputfilename
    except Exception as err:
        print(err)
    return False
