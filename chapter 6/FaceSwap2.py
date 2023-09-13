#Example 6.17a Face Swap with dlib and OpenCV
#Modified from:
#https://raw.githubusercontent.com/Jacen789/simple_faceswap/master/faceswap.py

#pip install opencv-python, dlib
#Download dlib face landmarks detection model：shape_predictor_68_face_landmarks.dat.bz2
#and unzip it
#http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


import os
import cv2
import dlib
import numpy as np
from FaceSwap_Utils import * # see Example 6.17b

here = os.path.dirname(os.path.abspath(__file__))

models_folder_path = os.path.join(here)  
faces_folder_path = os.path.join(here)  
predictor_path = os.path.join(models_folder_path, 'shape_predictor_68_face_landmarks.dat')  
image_face_path = os.path.join(faces_folder_path, 'face1.jpeg')  

detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor(predictor_path)  



def main():
    im1 = cv2.imread(image_face_path)  
    im1 = cv2.resize(im1, (600, im1.shape[0] * 600 // im1.shape[1]))
    landmarks1 = get_face_landmarks(im1, detector, predictor)  # 68_face_landmarks
    if landmarks1 is None:
        print('{}:Face not detected'.format(image_face_path))
        exit(1)
    im1_size = get_image_size(im1)  
    im1_mask = get_face_mask(im1_size, landmarks1)  

    cam = cv2.VideoCapture(0)
    while True:
        ret_val, im2 = cam.read()  # camera_image
        landmarks2 = get_face_landmarks(im2, detector, predictor)  # 68_face_landmarks
        if landmarks2 is not None:
            im2_size = get_image_size(im2)  
            im2_mask = get_face_mask(im2_size, landmarks2)  

            affine_im1 = get_affine_image(im1, im2, landmarks1, landmarks2)  
            affine_im1_mask = get_affine_image(im1_mask, im2, landmarks1, landmarks2)  

            union_mask = get_mask_union(im2_mask, affine_im1_mask)  


            affine_im1 = skin_color_adjustment(affine_im1, im2, mask=union_mask)  
            point = get_mask_center_point(affine_im1_mask)  
            seamless_im = cv2.seamlessClone(affine_im1, im2, mask=union_mask, p=point, flags=cv2.NORMAL_CLONE)  
            cv2.imshow('seamless_im', seamless_im)
        else:
            cv2.imshow('seamless_im', im2)
        if cv2.waitKey(1) == 27:  
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
