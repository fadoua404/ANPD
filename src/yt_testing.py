import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageOps

def add_border(input_image, output_image, border):
    img = Image.open(input_image)
 
    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border)
    else:
        raise RuntimeError('border is not an integer or tuple!')
 
    bimg.save(output_image)
 



pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

cascade=cv2.CascadeClassifier("src\data\haarcascade_russian_plate_number.xml")



def extract_num(img_name):
    global text
    img=cv2.imread(img_name)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate=cascade.detectMultiScale(gray,1.1,4) #number-plate detection

    #cropping the plate
    for (x,y,w,h) in  nplate :
        a,b=(int(0.02*img.shape[0]) ,int(0.025*img.shape[1]))
        cropped_img=img[y+a:y+h-a , x+b:x+w-b, :]

        #image processing (on the cropped image)
        kernel=np.ones((1,1) , np.uint8)
        cropped_img=cv2.dilate(cropped_img ,kernel,iterations=1)
        cropped_img=cv2.erode(cropped_img,kernel,iterations=1)

        

        # convertion the plate image to the gray scale
        plate_gray=cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        #convertion the plate image to black&white only 1
        (thresh , cropped_img)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)

        cropped_img_loc= "plate.png"
        cv2.imwrite(cropped_img_loc,cropped_img)
        cv2.imshow("cropped image", cv2.imread(cropped_img_loc))
        #cv2.imshow("threshold-image" , plate)
        #cv2.imwrite("temp.png",plate)
        #adding a border 
        add_border(cropped_img_loc, output_image='cropped_img_bord.jpg',border=150)

        #convertion the plate image to a text 
        text=pytesseract.image_to_string('cropped_img_bord.jpg')
        #deleting spaces
        text=''.join(e for e in text if e.isalnum())
        print("number is : " , text)
        
        #drawing a rectangle around the plate
        cv2.rectangle(img , (x,y) , (x+w,y+h) ,(51,51,255), 2)
        cv2.rectangle(img , (x, y-40) ,(x+w,y) ,(51,51,255),-1)

        #put the text on the rectangle
        cv2.putText(img,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        #cv2.imshow("Plate",cropped_img)



        cv2.imshow("Result",img)
        cv2.imwrite("Result.jpg",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

extract_num('src\images\car_1.jpg')




        

        


        