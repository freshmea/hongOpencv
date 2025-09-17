
import cv2
import pytesseract
from PIL import Image

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
cap = cv2.VideoCapture(4)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(gray)
    print(text)
    print(pytesseract.image_to_string(frame, lang='kor'))  # For Korean text

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
