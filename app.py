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
            outputpath = 'C:/Users/SBK/Desktop/tr_sign_web/data/outputs/'+os.path.basename(image_file)
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
    '''if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=0)'''
    # -- End of Sidebar

    st.header('✋Nesne Tanıma Algoritması Kullanarak Türkçe İşaret Dili Tespit Etme')
    st.subheader('👈🏽Seçenekleri Seçiniz')
    st.sidebar.markdown("# Türkçe İşaret Dili Tanıma Projesi"
                        "## ÖZET:"
                        "**Günlük hayatta insanlar; fikirlerini, düşüncelerini ve yaşadıklarını çevrelerindeki insanlara iletmek için birbirleriyle etkileşirler. Aynı dili konuşan insanlar bazı durumlarda dil kullanmadan da birtakım işaretlerle iletişim kurabilmektedirler. İşitme engellilerin iletişim sağlamak amacıyla parmak, el, kol, yüz hareketlerini kullanarak oluşturduğu işaret dili, doğal bir iletişim aracıdır."
"Ne yazık ki ülkemizde işaret dilinin ihmal edilmesi; bu konuda sözlük, dil bilgisi, ders kitabı, yardımcı ders kitapları ve malzemelerinin hazırlanması, araştırmalar yapılması yeterince teşvik görmediği gibi ulusal işaret dilimizin yaygınlaştırılmasını da geciktirdi. Bütün bu olumsuzlukların yanında okulumuzda ve çevremizde yaptığımız araştırmalar sonucunda işaret dili bilgisine dair yüzdemizin düşük olduğunu gördük. Bu kapsamda öncelikle bu konuda farkındalığı arttırmak adına okulumuzda işaret dili kullanımına yönelik bir seminer düzenledik. Ardından yaptığımız gözlemlerde gördük ki öğrencilerimizin bu konuda ilgi alaka ve farkındalıkları seminer öncesine göre, seminer sonrasında artış gösterdi. Bunun üzerine işaret dili bilmeyen bir birey ile işitme engelli bir bireyin iletişimini nasıl kuvvetlendirebiliriz sorusu üzerine bir proje hayal etmeye başladık. Ve düşündük ki nesne tanıma artık günlük hayatın birçok alanına yerleşmiş durumda. Teknolojinin sürekli olarak gelişmesi insanoğlunun yaptığı birçok işin bilgisayarlar tarafından yapılmasını sağlamaktadır."
"İşaret dili engelli insanların iletişim kurmasında sözel ifadelerin kullanılması yerine bedensel ifadelerin kullanılması esasına dayanır. Burada önemli olan hangi işaretin ne anlama geldiğini bir değişmezlik çerçevesinde ele almaktır. Bunu da nesne tanıma sistemleri kullanarak başarmak mümkündür. Öncelikli olarak işaret dilinin temel hareketlerinin veritabanının oluşturulması gerekir. Burada dikkat edilmesi gereken nokta veritabanın büyüklüğüdür. Veritabanı ne çok küçük ne de çok büyük olmalıdır. Çünkü veritabanı kısıtlı kaldığında hareketin sınıflandırılması zorlaşacak. Çok büyük olduğunda ise karar verme süresi artacağı gibi çakışmalar meydana gelebilir.**")

    imageInput('cuda', datasrc)

if __name__ == '__main__':
    main()


