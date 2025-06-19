 # 1. Veri Hazırlama
Bu aşamada, orijinal DICOM görüntüleri ve etiket verileri gruplara ayrılarak, derin öğrenme modellerinde yaygın olarak kullanılan NIfTI formatına dönüştürülür.

## Klasörlerin Gruplandırılması:
DICOM dosyaları genellikle çok sayıda görüntü dosyasından oluşur. Bellek yönetimi ve işlem kolaylığı için, her hasta verisi 64 görüntülük alt klasörlere bölünür.

in_path: Orijinal DICOM görüntü ve etiket klasörleri (örn: D:/Task03_Liver/dicom_file/labels, D:/Task03_Liver/dicom_file/images)

out_path: 64 görüntülük DICOM gruplarının kaydedileceği yeni klasörler (örn: D:/Task03_Liver/dicom_groups/labels, D:/Task03_Liver/dicom_groups/images)
Her hasta için dosyalar, maksimum 64 dosya içeren alt klasörlere taşınır.

## DICOM’dan NIfTI Formatına Dönüşüm:
DICOM grupları, derin öğrenme uygulamalarında yaygın kullanılan NIfTI formatına dönüştürülür.

dicom2nifti kütüphanesi kullanılır.

Her alt klasördeki DICOM serisi .nii.gz uzantılı NIfTI dosyasına çevrilir.

Dönüştürülen dosyalar sırasıyla nifti_files/images ve nifti_files/labels klasörlerine kaydedilir.

## Etiket Verilerinin Kontrolü:
Oluşan NIfTI dosyalarındaki etiketlerin boş olup olmadığı kontrol edilir.

Her .nii.gz dosyası nibabel ile yüklenir.

İçerikteki eşsiz değerler (np.unique) incelenir.

Dosyada tek bir eşsiz değer (genellikle 0) varsa, bu dosyanın boş olduğu anlamına gelir ve uyarı verilir.

# 2. Veri Ön İşleme (Preprocessing)
Medikal görüntüler, MONAI kütüphanesinin gelişmiş transformları kullanılarak modelin ihtiyacına uygun hale getirilir.

## Veri Seti Yapısı:

/path_to_data/
├── TrainVolumes/       # Eğitim görüntüleri
├── TrainSegmentation/  # Eğitim etiketleri
├── TestVolumes/        # Doğrulama görüntüleri
└── TestSegmentation/   # Doğrulama etiketleri

## prepare Fonksiyonu:

Veri dizinindeki dosyalar eşleştirilir ve listeler oluşturulur.

Eğitim ve doğrulama için farklı transform zincirleri uygulanır.

Kanal ekseni düzenlenir (EnsureChannelFirstD).

Voxel boyutları yeniden örneklenir (Spacingd).

Görüntü yoğunlukları normalize edilir (ScaleIntensityRanged).

Ön plan (karaciğer bölgesi) kırpılır (CropForegroundd).

Görüntüler hedef boyuta göre kırpılır veya pad edilir (ResizeWithPadOrCropd).

Son olarak PyTorch tensörlerine dönüştürülür (ToTensord).

## Cache Kullanımı:
cache=True seçeneğiyle veriler belleğe önceden yüklenip eğitim süreci hızlandırılır. Değilse, veriler çağrıldıkça işlenir.

# Yardımcı Fonksiyonlar (Utilities)
dice_metric(predicted, target)
Segmentasyon performansını Dice skoruyla ölçer.

calculate_weigths(val1, val2)
Azınlık sınıfın etkisini artırmak için arka plan ve ön plan piksel sayılarına göre ağırlık hesaplar.

train(model, data_in, loss, optim, max_epochs, model_dir, ...)
Modeli belirlenen epoch sayısı boyunca eğitir, performans metriklerini hesaplar, en iyi modeli kaydeder.

show_patient(data, SLICE_NUMBER=1, train=True, test=False)
Belirtilen slice üzerinden görüntü ve segmentasyon maskesini yan yana görselleştirir.

calculate_pixels(data)
Veri setindeki etiket maskelerinde arka plan ve ön plan piksel sayılarını toplar.

# Model Eğitimi
Projede, 3D karaciğer segmentasyonu için MONAI’dan UNet mimarisi kullanılmıştır.

Veri prepare fonksiyonuyla ön işleme tabi tutulup uygun DataLoader oluşturulur.

Model, tek kanal giriş (grayscale), iki sınıf çıkış (arka plan ve karaciğer) şeklindedir.

DiceLoss kullanılarak segmentasyon doğruluğu optimize edilir.

Adam optimizasyon algoritması düşük öğrenme oranı ve ağırlık çürümesi ile kullanılır.

Otomatik cihaz seçimi yapılır (GPU varsa CUDA, yoksa CPU).

Eğitim 200 epoch boyunca train fonksiyonu ile gerçekleştirilir, sonuçlar model dizinine kaydedilir.

# Test Aşaması
Model eğitildikten sonra test verisi üzerinde değerlendirilir. Performans metrikleri hesaplanır ve segmentasyon sonuçları görselleştirilir.
