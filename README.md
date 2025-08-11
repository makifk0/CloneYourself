# CloneYourself – Kişisel Konuşma ve Hafıza Sistemi

Bu proje, **kişisel bir yapay zeka sohbet asistanı** olan **Akif(Kendi isminizi verebilirsiniz)**’i simüle eder.  
Akif, kullanıcıyı tanımak için **akıllıca sorular sorar**, verilen cevapları analiz eder ve **kısa süreli (STM)**, **uzun süreli (LTM)** ve **güven (Confidence)** bellek sistemlerinde saklar.  
Ayrıca hava durumu, tarih bilgisi ve konuşma bağlamı gibi ek verileri kullanarak daha **doğal bir sohbet deneyimi** sunar.

---

## 🚀 Özellikler

- **Setup Modu**:  
  Kullanıcı hakkında **tekrarsız ve mantıklı sorular** sorar.  
  Cevaplardan elde ettiği bilgileri [accept] etiketleri ile işaretleyerek belleğe kaydeder.

- **Chat Modu**:  
  Gerçek bir kişi gibi davranır, sadece sorulan soruya **kısa ve net cevaplar** verir.  
  Konu takibi yapar, 5 dakika sonra konu değişirse yeni konuya geçer.

- **Bellek Yönetimi**:  
  - **STM (Short Term Memory)** – Güncel ve geçici bilgiler  
  - **LTM (Long Term Memory)** – Kişisel özellikler, tercihler, hobiler  
  - **Confidence** – Belirsiz, tahmini bilgiler  

- **ChromaDB Entegrasyonu** – Hafızayı vektör tabanlı olarak saklar ve arar.

- **OpenWeatherMap API Entegrasyonu** – Hava durumu bilgisini alır.

- **Konu Analizi** – Sohbet geçmişini inceleyerek mevcut konuyu belirler.

---

## 📦 Gereksinimler

- Python 3.9+
- Aşağıdaki kütüphaneler:
  ```bash
  pip install requests chromadb
