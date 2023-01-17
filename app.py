import streamlit as st
import torch
import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time




def imageInput(device, src):
    if src == 'Kendim görüntü yükleyeceğim':
        image_file = st.file_uploader("Bir görüntü yükleyiniz", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Yüklenen Görüntü', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath =  os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # call Model prediction--
           
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt', force_reload=True)
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # --Display predicton

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='MODEL ÇIKTISI', use_column_width='always')

    elif src == 'Test veri kümesi':
        # Image selector slider
        test_images = os.listdir('data/images/')
        test_image = st.selectbox('Lütfen bir görüntü seçiniz.', test_images)
        image_file = 'data/images/' + test_image
        submit = st.button("BAŞLAT!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Seçilen Görüntü', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                # call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt', force_reload=True)
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(r'C:\Users\SBK\Desktop\tr_sign_web\data\outputs\\'+os.path.basename(image_file))
                    # --Display predicton
                    img_ = Image.open(os.path.join(r'C:\Users\SBK\Desktop\tr_sign_web\data\outputs\\'+os.path.basename(image_file)))
                    st.image(img_, caption='MODEL ÇIKTISI')





def main():
    # -- Sidebar
    st.sidebar.title('⚙️Seçenekler')
    datasrc = st.sidebar.radio("Görüntüyü nereden seçeceğinizi giriniz.", ['Test veri kümesi', 'Kendim görüntü yükleyeceğim'])

    # option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    #if torch.cuda.is_available():
    #   deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=1)
    #else:
    #    deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=0)
    # -- End of Sidebar

    st.header('✋Nesne Tanıma Algoritması Kullanarak Türkçe İşaret Dili Tespit Etme')
    st.subheader('👈🏽Seçenekleri Seçiniz')
    st.sidebar.markdown("Detaylı Bilgi:"
        "https://github.com/denizerogluu/tr_sign_web#readme")

    imageInput('cuda', datasrc)

if __name__ == '__main__':
    main()


