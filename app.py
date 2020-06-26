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

def detect_faces(gray,img):
    #detect faces
    faces = face_cascade.detectMultiScale(gray,1.1, 4)
    #Draw rectangle
    for (x,y,w,h) in faces:
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
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image,use_column_width=True)
            st.write(type(our_image))

        enhance_type = st.sidebar.radio("Enhance Type", ["Original","Gray-Scale","Brightness", "Contrast"])
        if enhance_type == "Gray-Scale":
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img,1)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #st.write(new_img)
            st.image(gray,use_column_width=True)
            
        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider("Contrast",0.5,3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output,use_column_width=True)

        if enhance_type == 'Brightness':
            c_rate = st.sidebar.slider("Brightness",0.5,3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output, caption="Output",use_column_width=True)
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
                result_img = detect_faces(gray,img)
                st.image(result_img,use_column_width=True)
                #st.success("found {} faces".format(result_faces))
                

    elif choice == "About":
        st.subheader("About")

if __name__ == '__main__':
    main()  