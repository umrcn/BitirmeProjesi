# Derin Öğrenme Kullanarak Deri ve Beyin Tümörü Tespiti

Bu proje, dermoskopik görüntüler ve MRI taramaları kullanarak deri ve beyin tümörlerini tespit etmek için bir web uygulaması geliştirme üzerine odaklanmıştır. 

![Örnek Görüntü](https://user-images.githubusercontent.com/66179774/148924061-d8f462f9-cb2b-4699-b9f4-b7f92f461ae5.png)

## Kullanılan Diller & Çerçeveler

- Python
- JavaScript
- HTML
- CSS
- Bootstrap
- Flask
- sklearn
- TensorFlow
- Pillow
- OpenCV

## Kurulum

### Adım 1: Veri Setini İndirin ve Ayıklayın

- [BEYİN MRI TARAMALARI](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- [DERİ DERMOSKOPİK GÖRÜNTÜLERİ](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign)

### Adım 2: Projeyi İndirin ve Açın

### Adım 3: Gerekli Kütüphaneleri Yükleyin

```bash
pip install opencv-python
pip install pillow
pip install sklearn
pip install tensorflow
pip install flask
```
### Adım 4: Dosya Yollarını Güncelleyin

Ana eğitim dosyalarındaki dosya yollarını güncelleyin:

    maintrain_brain.py dosyasındaki MRI taramaları için dosya yolu
    mainTrain.py dosyasındaki deri kanseri görüntüleri için dosya yolu

Dizin yolu belirtirken, ilerleyen eğik çizgi ("/") kullanmayı unutmayın.
###  Adım 5: Modeli Eğitin

Ana eğitim dosyalarını sırayla çalıştırarak modeli eğitin.
###  Adım 6: Uygulamayı Çalıştırın

App.py dosyasını çalıştırarak uygulamayı başlatın. Model, yerel bir sunucuda yüklenecektir.
Nasıl Kullanılır

Uygulamayı çalıştırdıktan sonra, test görüntülerini yükleyerek sonuçları alabilirsiniz.
