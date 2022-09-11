

import numpy as np
from PIL import Image
from fastapi import UploadFile
import cv2
import numpy as np
from PIL import Image
import os
import dlib

async def watermarking_async(_image:UploadFile,_logo:UploadFile):
    
    image = Image.open(_image.file)
    logo = Image.open(_logo.file)
    
    #image.resize((150,150))
    #logo.resize((150,150))
    image_logow = np.array(image.convert('RGB'))
    h_image, w_image, _ = image_logow.shape
    
    logo = np.array(logo.convert('RGB'))
    h_logo, w_logo, _ = logo.shape
 
    center_y = int(h_image / 2)
    center_x = int(w_image / 2)
    top_y = center_y - int(h_logo / 2)
    left_x = center_x - int(w_logo / 2)
    bottom_y = top_y + h_logo
    right_x = left_x + w_logo

    roi = image_logow[top_y: bottom_y, left_x: right_x]
    result = cv2.addWeighted(roi, 1, logo, 1, 0)
    #cv2.line(image_logow, (0, center_y), (left_x, center_y), (0, 0, 255), 1)
    #cv2.line(image_logow, (right_x, center_y), (w_image, center_y), (0, 0, 255), 1)
    #image_logow[top_y: bottom_y, left_x: right_x] = result
    #img = Image.fromarray(image_logow, 'RGB')
    res, im_png = cv2.imencode(".png", result)
    return im_png


async def textmark_async(_image:UploadFile,text:str) -> bytes:

    image = Image.open(_image.file)
    image_logow = np.array(image.convert('RGB'))
    h_image, w_image, _ = image_logow.shape
    
    #logo = Image.open(_logo.file)
    #logo = np.array(logo.convert('RGB'))
 
    cv2.putText(image_logow, text=text, org=(w_image - 295, h_image - 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,
    color=(0,0,255), thickness=2, lineType=cv2.LINE_4); 
    res, im_png = cv2.imencode(".png", image_logow)
    return im_png
    #timg = Image.fromarray(image_logow, 'RGB')            
    #return timg.tobytes()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shape_predictor_68_face_landmarks = base_dir+r'\creative\AI\shape_predictor_68_face_landmarks.dat'


async def get_array_img(img):
    image = Image.open(img)
    image = image.resize((300,300))
    image_arr = np.array(image)
    return image_arr

async def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

async def face_swap_async(face1:UploadFile,face2:UploadFile):
    source_upload = face1.file
    destination_upload = face2.file
    #destination_upload.resize((300,300))
    #source_upload.resize((300,300))
    
    img = await get_array_img(source_upload)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros_like(img_gray)
    
    img2 = await get_array_img(destination_upload)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_68_face_landmarks)
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)	
    
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))	
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        
        cv2.fillConvexPoly(mask, convexhull, 255)	
        face_image_1 = cv2.bitwise_and(img, img, mask=mask)	
        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)	
        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])	
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = await extract_index_nparray(index_pt1)	
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = await extract_index_nparray(index_pt2)	
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = await extract_index_nparray(index_pt3)	
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)	
    # Face 2
    faces2 = detector(img2_gray)
    for face in faces2:
        landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))
            
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)
    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
    for triangle_index in indexes_triangles:# Triangulation of both faces
        tr1_pt1 = landmarks_points[triangle_index[0]]# Triangulation of the first face
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        
        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                            [tr1_pt2[0] - x, tr1_pt2[1] - y],
                            [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
        
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
        
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)# Lines space
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)
        
        tr2_pt1 = landmarks_points2[triangle_index[0]]# Triangulation of second face
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
        
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
        
        points = np.float32(points)# Warp triangles
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
        
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]# Reconstructing destination face
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
        
        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
        
    img2_face_mask = np.zeros_like(img2_gray)# Face swapped (putting 1st face into 2nd face)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    
    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)
    
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    
    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    resultimg = Image.fromarray(seamlessclone, 'RGB')
    res, im_png = cv2.imencode(".png", np.array(resultimg))
    return im_png
    #
    #return resultimg.tobytes()