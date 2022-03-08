import cv2
import numpy as np
import matplotlib.pyplot as plt

from tag_detector import *
from tag_processor import *
from homography_utils import *


if __name__ == "__main__":
    img_color = cv2.imread("assets/image_5.png")
    img_color = cv2.cvtColor(img_color , cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # detecting the tag from the image
    detector = TagDetector()
    detector.set_tag_image(img_gray)
    tag_corners = detector.detect_tag_corners()    

    # isolating the tag
    desired_corners = [(0,0), (200,0), (0,200), (200, 200)]
    H_matrix = compute_tag_corners_homography_matrix(tag_corners, desired_corners)
    isolated_tag_image = inverse_warp_image(H_matrix, img_gray, (200, 200))
    _, isolated_tag_image = cv2.threshold(isolated_tag_image, 127, 255, cv2.THRESH_BINARY) #performing thrsholding    

    # processing the tag image
    tag_processor = TagProcessor()
    tag_processor.set_tag_image(isolated_tag_image)
    tag_id, tag_orientaion = tag_processor.decode_tag()

    # superimposing the template image
    template_image_colored = cv2.imread("assets/testudo.png")
    template_image_corners = [(0,0), (0,300), (295, 300), (295,0)]

    H_matrix_2 = compute_tag_corners_homography_matrix(tag_corners, template_image_corners)

    if tag_orientaion == "down":        
        template_image_colored_rotated = np.rot90(template_image_colored, 1)
    elif tag_orientaion == "right":        
        template_image_colored_rotated = np.rot90(template_image_colored, 2)
    elif tag_orientaion == "left":
        template_image_colored_rotated = np.rot90(template_image_colored, 3)
        
    superimpose_image(H_matrix_2, img_color, template_image_colored_rotated)


    plt.figure(figsize=(50,50)) 
    plt.subplot(2, 2, 1), plt.imshow(img_gray, cmap='gray') 
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(isolated_tag_image, cmap='gray') 
    plt.title('Isolated tag Image - ID  '+ str(tag_id)+ '  Orientation  '+ str(tag_orientaion)), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(template_image_colored) 
    plt.title('Template image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(img_color) 
    plt.title('Superimposed result image'), plt.xticks([]), plt.yticks([])
    plt.show()