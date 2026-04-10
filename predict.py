from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("model.h5")

def predict_image(img_path):
    orig_image = cv2.imread(img_path)
    image = cv2.resize(orig_image, (128,128))
    image = image / 255.0
    image = image.reshape(1,128,128,3)

    result = model.predict(image)

    if result[0][0] > 0.5:
        gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        
        # Isolate the brain to avoid marking the skull
        _, head_mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        head_mask = cv2.erode(head_mask, kernel, iterations=4)
        cnts, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            brain = max(cnts, key=cv2.contourArea)
            brain_mask = np.zeros_like(gray)
            cv2.drawContours(brain_mask, [brain], -1, 255, -1)
            brain_mask = cv2.dilate(brain_mask, kernel, iterations=2)
            brain_only = cv2.bitwise_and(gray, gray, mask=brain_mask)
        else:
            brain_only = gray
            
        # Find tumor in the brain
        blurred = cv2.GaussianBlur(brain_only, (5, 5), 0)
        thresh = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        marked_img = orig_image.copy()
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            # Draw the exact contour of the tumor in green with reduced thickness
            cv2.drawContours(marked_img, [c], -1, (0, 255, 0), 2)
            
        marked_img = cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)
        return "Tumor Detected 🧠", marked_img
    else:
        return "No Tumor ✅", None

# Test
# print(predict_image("test.jpg"))