import cv2
import numpy as np
import matplotlib.pyplot as plt

from tag_detector import *
from tag_processor import *
from homography_utils import *

if __name__ == "__main__":
    vid_cap = cv2.VideoCapture("assets/tagvideo.mp4")
    detector = TagDetector()
    tag_processor = TagProcessor()

    desired_tag_img_size = 200
    desired_corners = [(0,0), (desired_tag_img_size,0), (0,desired_tag_img_size), (desired_tag_img_size, desired_tag_img_size)]

    # superimposing the template image
    template_image = cv2.imread("assets/testudo.png")
    template_image_corners = [(0,0), (0,template_image.shape[1]),  (template_image.shape[0], template_image.shape[1]), (template_image.shape[0],0)]

    frame_counter = 0
    while True:
        _, frame = vid_cap.read()
        img_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_counter % 5 == 0:

            detector.set_tag_image(img_grayscale)
            tag_corners = detector.detect_tag_corners()              

            if len(tag_corners) == 4:  
                
                try:
                    H_matrix = compute_tag_corners_homography_matrix(tag_corners, desired_corners)
                    isolated_tag_image = inverse_warp_image(H_matrix, img_grayscale, (desired_tag_img_size, desired_tag_img_size))
                    _, isolated_tag_image = cv2.threshold(isolated_tag_image, 127, 255, cv2.THRESH_BINARY) #performing thrsholding    

                    # processing the tag image
                    tag_processor.set_tag_image(isolated_tag_image)
                    tag_id, tag_orientaion = tag_processor.decode_tag() 

                    if tag_orientaion == "down":        
                        template_image_rotated = np.rot90(template_image, 3)
                    elif tag_orientaion == "right":        
                        template_image_rotated = np.rot90(template_image, 2)
                    elif tag_orientaion == "left":
                        template_image_rotated = np.rot90(template_image, 0)
                    elif tag_orientaion == "up":
                        template_image_rotated = np.rot90(template_image, 1)

                    H_matrix_2 = compute_tag_corners_homography_matrix(tag_corners, template_image_corners)  
                    superimpose_image(H_matrix_2, frame, template_image_rotated)
                except:
                    pass

            for corner in tag_corners:
                cv2.circle(frame, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 5)


            cv2.imshow("canny_edge", frame)

            if cv2.waitKey(20) == ord("q"):
                break
        
        frame_counter+= 1

    vid_cap.release()
    cv2.destroyAllWindows()