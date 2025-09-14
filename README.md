 # CudaMultiGPU Documentation

[English Version](README_EN.md) | [Turkish Version](README.md)

---

## 📖 About
This document explains step by step how to use the **CudaMultiGPU** class in your projects.

- Turkish documentation: [README.md](README.md)  
- English documentation: [README_EN.md](README_EN.md)

---


# CudaMultiGPU Kullanım Dokümanı

Bu doküman, projelerinizde **CudaMultiGPU** sınıfının nasıl kullanılacağını adım adım anlatmaktadır.

---

## 1. Sınıfı Projeye Ekleme

`gpu_utils.py` dosyasını proje kök dizininize kopyalayın.  
Kodunuzda sınıfı içe aktarın:

<img width="468" height="45" alt="image" src="https://github.com/user-attachments/assets/98f5ad01-c035-44d1-b97a-30abb18883ab" />


```python
from gpu_utils import CudaMultiGPU
```

---

## 2. DEVICE Tanımlamasını Kaldır

Kodunuzdan `DEVICE` tanımını kaldırın.

<img width="468" height="46" alt="image" src="https://github.com/user-attachments/assets/96a89fe6-c05b-49d7-94e3-b2fe274fe525" />


---

## 3. Modeli Sarmalama

Mevcut PyTorch modelinizi çoklu GPU desteğiyle kullanmak için:

<img width="468" height="51" alt="image" src="https://github.com/user-attachments/assets/bacfc348-752e-4fc6-9bd8-7fbaa56a91fc" />


```python
model = MyModel()              # mevcut PyTorch modeliniz
model = CudaMultiGPU(model)    # tüm CUDA aygıtlarını algılar ve DataParallel ile sarar
DEVICE = model.device          # verilerin taşınacağı aygıt
```

- CUDA aygıtı yoksa model otomatik olarak **CPU** üzerinde çalışır.  
- `CudaMultiGPU` örneği çağrılabilir olduğundan şu şekilde kullanılabilir:  

```python
outputs = model(input_tensor)
```

- Altta yatan modele doğrudan erişmek için:

```python
model.model
```

---

## 4. Optimizasyonu Ayarlama

`CudaMultiGPU`, alttaki modele özellik ve metotları ilettiği için parametrelere doğrudan erişilebilir:

<img width="468" height="28" alt="image" src="https://github.com/user-attachments/assets/d590b533-1c83-4bc5-95c7-63ec67d224b5" />


```python
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
```

---

## 5. Eğitim Döngüsünde Yapılacak Değişiklikler

<img width="468" height="77" alt="image" src="https://github.com/user-attachments/assets/259c951c-3875-4c90-8e6b-7934eb350319" />


### Verileri Doğru Aygıta Taşıma
```python
inputs  = inputs.to(DEVICE)
targets = targets.to(DEVICE)
```

### Modeli Çağırma
```python
outputs = model(inputs)
```

### Loss Ortalaması
`DataParallel`, her GPU için ayrı loss döndürebilir.  
Geri yayılım için ortalama alınmalıdır:

```python
loss = outputs.loss.mean()  # veya benzer şekilde hesaplanan loss.mean()
```

### Geri Yayılım ve Optimizasyon
```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

## 6. Batch Size ve Num Workers Ayarlama

<img width="297" height="152" alt="image" src="https://github.com/user-attachments/assets/057fd002-eb19-4de8-bf8b-2bb5f9dbe1f0" />


Bilgisayarda kullanılan ekran kartı sayısına ve yapılan deneyin işlem kapasitesine göre `batch_size` değeri **128–256** aralığında yükseltilebilir.  

**num_workers** için önerilen hesaplama şu şekildedir:

```
num_workers = CPU çekirdek sayısı / GPU sayısı
```

### Pratik Öneriler:
- 16 çekirdekli CPU ve 3 GPU → `num_workers = 4–6`  
- 32 çekirdekli CPU ve 3 GPU → `num_workers = 8–12`  
- 64 çekirdekli CPU (Threadripper / EPYC gibi) → `num_workers = 16+`  

GPU kullanımı gözlemlenerek değerler gerektiğinde daha da yükseltilebilir.

---

## Lisans
Bu doküman ve `CudaMultiGPU` sınıfı, kendi projelerinizde özgürce kullanılabilir.
