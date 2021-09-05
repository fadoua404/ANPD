import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img=cv2.imread("src\images\test.png")
text=pytesseract.image_to_string(img)
    
text=''.join(e for e in text if e.isalnum())

print(text)