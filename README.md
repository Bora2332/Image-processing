## Veri Hazırlama

Bu bölümde, orijinal DICOM görüntüleri ve etiket verileri gruplara ayrılarak, derin öğrenme modellerinde yaygın olarak kullanılan **NIfTI** formatına dönüştürülmektedir.

### 1. Klasörlerin Gruplandırılması

DICOM dosyaları genellikle çok sayıda görüntü dosyasından oluşur. Bellek yönetimi ve işlem kolaylığı için, her hasta verisi 64 görüntülük alt klasörlere bölünmektedir.

- `in_path` : Orijinal DICOM görüntü ve etiket klasörleri (örn: `D:/Task03_Liver/dicom_file/labels`, `D:/Task03_Liver/dicom_file/images`)
- `out_path`: 64 görüntülük DICOM gruplarının kaydedileceği yeni klasörler (örn: `D:/Task03_Liver/dicom_groups/labels`, `D:/Task03_Liver/dicom_groups/images`)

Her hasta için dosyalar, her biri maksimum 64 dosya içeren alt klasörlere taşınır.

### 2. DICOM’dan NIfTI Formatına Dönüşüm

DICOM grupları, derin öğrenme uygulamalarında yaygın kullanılan NIfTI formatına çevrilir.

- `dicom2nifti` kütüphanesi kullanılır.
- Her alt klasördeki DICOM serisi `.nii.gz` uzantılı NIfTI dosyasına dönüştürülür.
- Dönüştürülen dosyalar sırasıyla `nifti_files/images` ve `nifti_files/labels` klasörlerine kaydedilir.

### 3. Etiket Verilerinin Kontrolü

Dönüştürülen NIfTI dosyalarındaki etiketlerin boş olup olmadığı kontrol edilir.

- Her `.nii.gz` dosyası `nibabel` ile yüklenir.
- Veri matrisi alınır ve içerisindeki eşsiz değerler (`np.unique`) incelenir.
- Eğer dosyada sadece tek bir eşsiz değer varsa (genellikle 0), bu dosyanın boş olduğu anlamına gelir ve uyarı mesajı yazdırılır.

### 4. Gereksinimler

Bu işlemler için aşağıdaki Python paketleri gereklidir:

```bash
pip install glob2 pytest-shutil pydicom==2.3.1 dicom2nifti==2.4.6 nibabel numpy
## Veri Ön İşleme (Preprocessing)

Bu projede kullanılan medikal görüntüler, MONAI kütüphanesinin güçlü transformları ile ön işleme tabi tutulmaktadır. Bu sayede modeller için uygun boyut, ölçek ve formatta veriler sağlanır.

### 1. Veri Seti Yapısı

- Eğitim ve doğrulama görüntüleri NIfTI (`.nii.gz`) formatındadır.
- Veri dizininde aşağıdaki klasör yapısı vardır:

/path_to_data/
├── TrainVolumes/ # Eğitim görüntüleri
├── TrainSegmentation/ # Eğitim etiketleri
├── TestVolumes/ # Doğrulama görüntüleri
└── TestSegmentation/ # Doğrulama etiketleri

### 2. `prepare` Fonksiyonu

Bu fonksiyon, belirtilen veri dizininden dosyaları okur ve aşağıdaki işlemleri uygular:

- Dosya isimlerini eşleştirerek veri listeleri oluşturur.
- Veri setine göre farklı transform zincirleri kullanır (eğitim ve doğrulama için ayrı).
- Görüntüler ve etiketler için kanal eksenli düzenleme yapılır (`EnsureChannelFirstD`).
- Voxel boyutları (`pixdim`) belirtilen değere yeniden örneklenir (`Spacingd`).
- Görüntü yoğunlukları belirlenen aralığa ölçeklenir (`ScaleIntensityRanged`).
- Görüntülerdeki ön plan (karaciğer bölgesi) kırpılır (`CropForegroundd`).
- Görüntü ve etiketler istenilen `spatial_size` boyutuna göre kırpılır veya pad edilir (`ResizeWithPadOrCropd`).
- Son olarak, veriler PyTorch tensörlerine dönüştürülür (`ToTensord`).

```python
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstD(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=pixdim),
    ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
    ToTensord(keys=["image", "label"])
])
3. Cache Kullanımı
cache=True parametresi verilirse, CacheDataset kullanılarak veriler belleğe önceden yüklenir ve eğitim süreci hızlanır.

Değilse, Dataset ile her çağrıda veriler transform edilerek yüklenir.

### Utilities partı
dice_metric(predicted, target)
Modelin segmentasyon performansını ölçmek için Dice skoru hesaplar. Tahmin edilen maskeyle gerçek etiketi karşılaştırarak doğruluğu 0 ile 1 arasında bir değer olarak verir.

calculate_weigths(val1, val2)
Veri setindeki arka plan ve ön plan piksel sayılarına göre sınıf ağırlıklarını dengeler. Böylece azınlık sınıfın etkisi artırılarak eğitim dengeli hale getirilir.

train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1, device=torch.device("cuda:0"))
Modeli belirtilen epoch sayısı boyunca eğitir. Eğitim ve doğrulama verisi üzerinde kayıp ve Dice skorlarını hesaplar, en iyi modeli kayıt eder. GPU desteği mevcuttur.

show_patient(data, SLICE_NUMBER=1, train=True, test=False)
Veri setinden bir hastanın belirli bir dilimini (slice) görselleştirir. Hem görüntüyü hem segmentasyon maskesini yan yana göstererek sonuçları hızlıca incelemeyi sağlar.

calculate_pixels(data)
Veri setindeki tüm etiket maskelerinde arka plan ve ön plan piksel sayılarını toplar. Veri dengesizliği analizi ve ağırlıklandırma için temel veri sağlar.

### Train partı
Bu proje kapsamında, 3 boyutlu karaciğer segmentasyonu için MONAI kütüphanesinden UNet mimarisi kullanılmıştır. Öncelikle, prepare fonksiyonu ile medikal görüntüler ve etiketler ön işleme tabi tutularak uygun formatta ve boyutta veri yükleyiciler (DataLoader) oluşturulur. Model, 3D uzaysal veriler için tasarlanmış olup, girişte tek kanal (grayscale) görüntü kabul eder ve çıktı olarak iki sınıf (arka plan ve karaciğer) üretir. Eğitim sırasında DiceLoss fonksiyonu ile segmentasyon doğruluğu optimize edilir. Optimizasyon için Adam algoritması tercih edilmiş ve küçük öğrenme oranı ile ağırlık çürümesi (weight_decay) uygulanmıştır. Kodda otomatik olarak GPU varsa CUDA cihazı, yoksa CPU kullanılacak şekilde cihaz seçimi yapılmaktadır. Son olarak, tanımlanan model, veri ve kayıp fonksiyonu ile 200 epoch boyunca train fonksiyonu çağrılarak eğitilir; eğitim sonuçları belirtilen model dizinine kaydedilir.

### Testing partı

Burada modelin sonucu doğrultusunda testing aşamasını yaptım ve grafikleri gösterdim
