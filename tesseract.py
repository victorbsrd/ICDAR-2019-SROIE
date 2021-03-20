import cv2
import os,argparse
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#We then Construct an Argument Parser
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",
                required=True,
                help="Path to the image folder")
ap.add_argument("-p","--pre_processor",
                default="thresh",
                help="the preprocessor usage")
args=vars(ap.parse_args())

#We then read the image with text
images=cv2.imread(args["image"])

#convert to grayscale image
gray=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

#checking whether thresh or blur
if args["pre_processor"]=="thresh":
    cv2.threshold(gray, 0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
if args["pre_processor"]=="blur":
    cv2.medianBlur(gray, 3)

#memory usage with image i.e. adding image to memory
filename = "{}.jpg".format(os.getpid())
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
plt.imshow(images, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
