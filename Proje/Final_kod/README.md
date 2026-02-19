# Polip Segmentasyonu - Semi-Supervised Learning Projesi
# Kodun drive linki: https://drive.google.com/file/d/1O7fW1JLh7_SOG2xGlSMEzxozPucJPCbj/view?usp=sharing

Bu proje, kolon poliplerinin segmentasyonu için **yarı-gözetimli (semi-supervised) derin öğrenme** yaklaşımını uygulamaktadır. Proje, **Kvasir-SEG** veri seti üzerinde çalışmaktadır ve ResNet34 tabanlı bir U-Net benzeri mimari kullanmaktadır.

---

## 📁 Dosya Yapısı

| Dosya | Açıklama |
|-------|----------|
| `FinalCode.ipynb` | **Eğitim (Training)** notebook'u - Model eğitimi ve checkpoint kaydetme (Test için de kullanılabilir)|
| `FinalTest.ipynb` | **Test/Değerlendirme** notebook'u - Önceden eğitilmiş modeli yükleme ve test etme |

---

## 🏗️ Model Mimarisi

### Genel Yapı
- **Encoder:** ResNet34 (ImageNet pre-trained) tabanlı 5 seviyeli encoder
- **Segmentation Decoder:** Bilinear upsampling ile U-Net tarzı decoder
- **Inpainting Decoder:** ConvTranspose2d ile reconstruction decoder
- **GFF (Gated Feature Fusion):** Segmentasyon ve inpainting özelliklerini birleştiren modül
- **Deep Supervision:** Tüm decoder seviyelerinden side-output'lar

### Modül Detayları

| Modül | Açıklama |
|-------|----------|
| `EncoderBlock` | ResNet34'ün conv1, layer1-4 katmanlarını kullanır |
| `DecoderBlock` | Conv → BN → ReLU → Conv → BN → ReLU → Upsample |
| `GFF` | Reset ve Select gate'leri ile dual-channel fusion |
| `SideBlock` | Conv → Dropout(0.1) → Conv → Sigmoid |

---

## 📊 Veri Setleri

### 1. Kvasir-SEG Dataset

- **Kaynak:** [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)
- **Toplam Görüntü:** 1000 adet polip görüntüsü ve maske
- **Bölünme:**
  - Train: 600 görüntü (%60)
  - Validation: 200 görüntü (%20)
  - Test: 200 görüntü (%20)
- **Semi-Supervised Ayarı:**
  - Labeled: 120 görüntü (%20 of train)
  - Unlabeled: 480 görüntü (%80 of train)

### 2. ISBI 2016 Skin Lesion Dataset

- **Kaynak:** [ISBI 2016 Challenge - Skin Lesion Analysis](https://challenge.isic-archive.com/landing/2016/)
- **Toplam Görüntü:** 900 adet dermoskopik cilt lezyonu görüntüsü ve maske
- **Bölünme:**
  - Train: 540 görüntü (%60)
  - Validation: 180 görüntü (%20)
  - Test: 180 görüntü (%20)
- **Semi-Supervised Ayarı:**
  - Labeled: 108 görüntü (%20 of train)
  - Unlabeled: 432 görüntü (%80 of train)

---

## 🚀 Kullanım

### 1. FinalCode.ipynb (Eğitim)

Bu notebook modeli sıfırdan eğitmek için kullanılır.

#### Eğitim Parametreleri
```python
args = SimpleNamespace(
    batch_size = 4,
    lr = 1e-3,
    weight_decay = 1e-5,
    nEpoch = 80,
    power = 0.9
)
```

#### ⏱️ Eğitim Süresi
> **NVIDIA A100 GPU ile 100 epoch yaklaşık 40 dakika sürmektedir.**


---

### 2. FinalTest (1).ipynb (Test/Değerlendirme)

Bu notebook önceden eğitilmiş bir modeli yükleyerek test yapmak için kullanılır.

#### ⚠️ ÖNEMLİ: Checkpoint Dizini Ayarı

**Google Colab'da çalıştırırken:**

Eğer `best_model.pth` dosyasını Colab'a manuel olarak yüklediyseniz (Google Drive yerine):

```python
# Varsayılan ayar (Google Drive'dan yükleme):
checkpoint_dir = "/content/drive/MyDrive/models2"

# Manuel yükleme için DEĞİŞTİRİN:
checkpoint_dir = "/content"
```

**Adımlar:**
1. `best_model.pth` dosyasını Colab'a yükleyin
2. Notebook'ta ilgili hücreyi bulun (24. hücre civarı)
3. `checkpoint_dir` değişkenini `/content` olarak değiştirin
4. Hücreyi çalıştırın

---

## 📈 Loss Fonksiyonları

### Labeled Data (Segmentasyon)
- **BceDiceLoss:** Binary Cross Entropy + Dice Loss
- **Deep Supervision:** Tüm decoder seviyelerinde loss hesaplama

### Unlabeled Data (Inpainting)
- **L1 Loss:** Masked reconstruction loss
- Sadece segmentasyon maskesinin >0.5 olduğu bölgelerde hesaplanır

### Toplam Loss
```
Total Loss = 2 × Loss_labeled + Loss_unlabeled
```

---

## 📏 Değerlendirme Metrikleri

- **Dice Score (F1)**
- **IoU (Intersection over Union)**
- **Recall (Sensitivity)**
- **Specificity**
- **Precision**
- **F2 Score**
- **Accuracy**

---

## 💾 Model Checkpoint

Eğitim sırasında en iyi validation Dice score'una sahip model otomatik olarak kaydedilir:

```
./data/checkpoint/exp1/best_model.pth
```

---

## 🔧 Gereksinimler

```bash
pip install torch torchvision opencv-python scipy scikit-image tqdm albumentations
```

### GPU Gereksinimi
- CUDA destekli GPU önerilir
- Google Colab'da A100 GPU ile test edilmiştir

---

## 📝 Hücrelerin İşlevleri

### FinalCode.ipynb

| Hücre No | İşlev |
|----------|-------|
| 1-6 | Kurulum, veri indirme ve yükleme |
| 7-8 | Data augmentation ve Dataset sınıfı |
| 9-11 | Train/Valid/Test split |
| 12-14 | Veri görselleştirme |
| 20-21 | Model mimarisi (Encoder, Decoder, GFF, SideBlock) |
| 22 | Tam model (MyModel) |
| 28-29 | Loss fonksiyonları |
| 30 | Eğitim döngüsü |
| 31 | Google Drive mount ve checkpoint |
| 32-47 | Test ve değerlendirme |

### FinalTest (1).ipynb

| Hücre No | İşlev |
|----------|-------|
| 1-14 | Kurulum ve veri hazırlığı (FinalCode ile aynı) |
| 15-23 | Model ve loss tanımları |
| 24 | **Checkpoint yükleme (checkpoint_dir ayarı burada!)** |
| 25-28 | Test değerlendirmesi ve metrikler |

---

## 🔄 Çalıştırma Sırası

### Eğitim İçin (FinalCode.ipynb):
1. Tüm hücreleri sırayla çalıştırın
2. Eğitim tamamlandığında `best_model.pth` kaydedilir
3. Modeli Google Drive'a kaydedin (kalıcılık için)

### Test İçin (FinalTest (1).ipynb):
1. Veri hazırlığı hücrelerini çalıştırın (1-23)
2. **checkpoint_dir** ayarını kontrol edin
3. Modeli yükleyin ve test hücrelerini çalıştırın

---

## 📚 Referanslar

- **Zhang et al.** (2021). *Self-Supervised Correction Learning for Semi-Supervised Biomedical Image Segmentation.* MICCAI 2021.
- Kvasir-SEG Dataset: https://datasets.simula.no/kvasir-seg/
- ISBI 2016 Skin Lesion Challenge: https://challenge.isic-archive.com/landing/2016/

---