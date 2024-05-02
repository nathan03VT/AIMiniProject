import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read and resize an image, maintaining aspect ratio
def preprocessing(image_path, max_width=1000):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    # Resize if width exceeds specified maximum width while maintaining aspect ratio
    if w > max_width:
        new_w = max_width
        ar = w / h
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# Thresholding to create a binary image
def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    return thresh

def segment_lines(thresh_img):

    # Dilate to connect characters
    kernel = np.ones((3, 85), np.uint8)  
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)  
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    plt.imshow(cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB), cmap='gray')
    plt.show()

    return sorted_contours_lines

def segmentFilters(image, sorted_contours_lines):
    img_copy = image.copy()
    average_height = np.mean([cv2.boundingRect(ctr)[3] for ctr in sorted_contours_lines])  # Calculate average height of lines
    buffer_size = 4  

    for i, ctr in enumerate(sorted_contours_lines):
        x, y, w, h = cv2.boundingRect(ctr)

        # Extract region of interest (ROI) and convert to grayscale
        roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)  
        edges = cv2.Canny(roi_gray, 50, 150)
        num_edges = np.sum(edges > 0)
       
        # Filter out sections with very few edges
        if num_edges < 100:
            continue

        # Add buffer after filtering
        y -= buffer_size
        h += 2 * buffer_size

        # Filter out segments smaller than half the average height
        if h < 0.5 * average_height:
            continue

        # Ensure that the adjusted bounding box does not go beyond the image boundaries
        y = max(0, y)
        h = min(image.shape[0] - y, h)

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (40, 100, 250), 2)
        roi = image[y:y+h, x:x+w]
        yield roi

if __name__ == "__main__":
    image_path = 'handwriting.png'

    # Reading and resizing
    img = preprocessing(image_path)

    # Preprocessing (Thresholding)
    thresh_img = thresholding(img)
    plt.imshow(thresh_img, cmap='gray')
    plt.show()

    # Line segmentation
    sorted_contours_lines = segment_lines(thresh_img)

    #Line segmentation visualization
    for i, char_segment in enumerate(segmentFilters(img, sorted_contours_lines)):
        plt.imshow(char_segment)
        plt.title(f"Segmented section {i+1}")
        plt.pause(0.001)
        plt.draw()
        plt.waitforbuttonpress()
    plt.close()
