import streamlit as st
import cv2 
from PIL import Image,ImageEnhance, ExifTags
import numpy as np
import os


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect_features(gray,img,feature):
    #detect faces
    if feature == 'face':
        features = face_cascade.detectMultiScale(gray,1.1, 4)
    if feature == 'eyes':
        features = eyes_cascade.detectMultiScale(gray,1.1, 4)
    
    #Draw rectangle
    for (x,y,w,h) in features:
        # select the areas where the face was found
        roi_color = img[y:y+h, x:x+w]
        # blur the colored image
        blur = cv2.GaussianBlur(roi_color, (101,101), 0)        
        # Insert ROI back into image
        img[y:y+h, x:x+w] = blur            
    
    # return the blurred image
    return img


def main():
    """face detection app"""
    st.title("Protect-Ur-Data")
    st.text("Your data belongs to you, not a police algorithm. Select an image to blur faces")
    hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    activities = ["Detection" , " About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Detection':
        st.subheader("Face Detection")
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

        if image_file is not None:
            
            st.text("Original Image")
            '''
            try:
                #image=Image.open(filepath)
                our_image = Image.open(image_file)

                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation]=='Orientation':
                        break

                exif=dict(our_image._getexif().items())

                if exif[orientation] == 3:
                    our_image=our_image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    our_image=our_image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    our_image=our_image.rotate(90, expand=True)

                

            except (AttributeError, KeyError, IndexError):
                # cases: image don't have getexif
                our_image = Image.open(image_file)'''
            our_image = Image.open(image_file)
            st.image(our_image,use_column_width=True)
            st.write(type(our_image))
        else:
            if st.image is None:
                st.image(our_image, use_column_width=True)


        #face detection
        task = ["Faces","Eyes"]
        feature_choices = st.sidebar.selectbox("Find Features", task)

        if st.button("Process"):
            if feature_choices == "Faces":
                new_img = np.array(our_image.convert('RGB'))
                img = cv2.cvtColor(new_img,1)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                result_img = detect_features(gray,img,'face')
                st.image(result_img,use_column_width=True)
                #st.success("found {} faces".format(result_faces))
            
            if feature_choices == "Eyes":
                new_img = np.array(our_image.convert('RGB'))
                img = cv2.cvtColor(new_img,1)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                result_img = detect_features(gray,img,'eyes')
                st.image(result_img,use_column_width=True)

    elif choice == "About":
        st.subheader("About")

if __name__ == '__main__':
    main()  