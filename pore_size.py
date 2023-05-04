import cv2
import numpy as np

def pore_size(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image
    height, width = gray.shape
    gray = cv2.resize(gray, (width//4, height//4))

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    # Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Edge Detection
    edges = cv2.Canny(thresh, 1, 2)

    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    # Filter contours
    filtered_contours = []
    areas = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt) * 13500/height * 13500/height
        if cnt_area > 100:
            filtered_contours.append(cnt)
            areas.append(cnt_area)
    average_area = np.average(areas)
    biggest_pore = max(areas)
    # Fit Circles
    circles = []
    for cnt in filtered_contours:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        circles.append((int(x), int(y), int(radius)))

    # Pick the biggest circle
    if len(circles) == 0:
        return image, blurred, edges, average_area
        
    
    biggest_circle = max(circles, key=lambda x: x[2])
    x, y, r = biggest_circle

    # Get the center and the radius of the region of interest
    height, width = image.shape[:2]
    center = (width//2, height//2)
    radius = min(center[0], center[1])

    # Highlight the biggest pore in the region of interest
    if (x*4 + center[0] - radius) - r*4 >= 0 and (y*4 + center[1] - radius) - r*4 >= 0 and (x*4 + center[0] - radius) + r*4 <= 2*radius and (y*4 + center[1] - radius) + r*4 <= 2*radius:
        cv2.circle(image, (x*4 + center[0] - radius, y*4 + center[1] - radius), r*4, (0,255,0), 2)

    return image, blurred, edges, average_area, biggest_pore, areas
