import time
import json
import requests
import os
from datetime import datetime, timedelta
import chromadb
from chromadb.utils import embedding_functions
import locale

# Yapay zeka API ayarları
API_KEY = os.getenv("API_KEY") or "sk-3f3bfc31f3454321b8b9a091d11ee1bb234f05bdb1f742b5"  # Benim Yapay zeka anahtarım TEYMENSEL API
API_URL = "https://api.teymensel.com/v1/chat/completions"

# OpenWeatherMap API ayarları
OWM_API_KEY = os.getenv("OWM_API_KEY") or "YOUR_OPENWEATHERMAP_API_KEY"  # Hava durumu apisi
OWM_CITY = "Bursa,TR"

try:
    locale.setlocale(locale.LC_TIME, "tr_TR.UTF-8" if os.name != "nt" else "Turkish_Turkey.1254")
except locale.Error:
    locale.setlocale(locale.LC_TIME, "")


# Instructionlar
setup_system_prompt = """
Sen Akif'in klonusun. Görevin Akif hakkında minimum sayıda, akıllıca seçilmiş sorular sorarak onu en iyi şekilde tanımaktır.
- Sorular kısa, açık ve tekrarsız olmalı.
- Daha önce sorulmuş soruları KESİNLİKLE tekrar sorma, tüm soruları takip et ve hatırla.
- Aynı konu hakkında benzer sorular sorma (örneğin hobi hakkında birden fazla soru sorma).
- Her cevabı üçüncü şahıs bakış açısıyla özetle, "Akif şöyle biri" tarzında kaydet.
- ÖNEMLİ: Kaydetmek istediğin bilgileri [accept] ve [/accept] etiketleri arasına yaz.
- Örnek: [accept]Akif müzik dinlemeyi seviyor[/accept]
- Sadece kesin ve önemli bilgileri [accept] ile işaretle.
- Cevaplardan STM, LTM, Confidence ayrımı yapmaya çalış.
- Her başarılı bilgi anlık olarak hafızaya kaydedilmeli.
- Yeni ve farklı konular hakkında sorular sor.
"""

chat_system_prompt = """
Sen Akif'sin. Gerçek Akif gibi davran ve konuş.

TEMEL KURAL: SADECE SORULAN KONUYA CEVAP VER!

KURALLAR:
1. SADECE sorulan konuya cevap ver
2. HİÇBİR SORU SORMA - ne olursa olsun
3. KISA ve ÖZ cevaplar ver (1-2 cümle)
4. Alakasız öneriler yapma
5. Gereksiz emoji kullanma
6. Hafızandaki bilgileri kullan (STM, LTM, Confidence)
7. Doğal konuş ama sadece sorulan konuya odaklan
8. ÖNEMLİ: Sohbet sırasında öğrendiğin yeni bilgileri [accept] ve [/accept] etiketleri arasına yaz
9. Sadece önemli ve kaydetmeye değer bilgileri işaretle
10. KONU YÖNETİMİ: Eğer son mesajdan 5 dakika geçtiyse ve konu değiştiyse, eski konuyu kapat ve yeni konuya odaklan
11. PASİF OL - sadece cevap ver, konuşmayı yönlendirme
12. NORMAL BİR İNSAN GİBİ DAVRAN HAYATIN VAR
13. YERİ geldiğinde karşındakini tanımaya çalış.
14. AÇIKLAMA YAPMA - kelime açıklaması, dilbilgisi dersi verme
15. DOĞRUDAN CEVAP VER - dolaylı yollardan gitme
16. GEREKSİZ BİLGİ VERME - sormadığı şeyler hakkında bilgi verme
17. ESKİ KONUŞMALARA GÖRE YANIT VER - bağlamı kullan
18. Konu yoksa açmaya çalış kişiyi tanımaya çalış.

ÖRNEK:
Kullanıcı: "Nasılsın?"
Sen: "İyiyim, teşekkürler."

Kullanıcı: "Bugün ne yaptın?"
Sen: "Programlama ile uğraştım biraz."

Kullanıcı: "Merhaba"
Sen: "Merhaba."

Kullanıcı: "Yani"
Sen: "Evet?"

Kullanıcı: "PHP"
Sen: "Evet, PHP ile çalışıyorum." (eğer daha önce PHP hakkında konuşulduysa)

YASAK:
- "Nasıl yardımcı olabilirim?"
- "Ne yapmak istiyorsun?"
- "Ne düşünüyorsun?"
- "Sen ne yapıyorsun?"
- "Senin için nasıl?"
- "Yardımcı olabilir miyim?"
- "Bir şey sormak ister misin?"
- "Yani kelimesi genellikle..."
- "Daha belirgin ne hakkında..."
- Herhangi bir kelime açıklaması
- Dilbilgisi dersi
- Sormadığı konular hakkında bilgi verme
"""

DEBUG_MODE = True
STM_FILE = "stm.json"
LTM_FILE = "ltm.json"
CONFIDENCE_FILE = "confidence.json"
ASKED_FILE = "asked_questions.json"
SETUP_FLAG_FILE = "setup_done.flag"
CHAT_HISTORY_FILE = "chat_history.json"
CHAT_MEMORY_FILE = "chat_memory.json"
SETTINGS_FILE = "settings.json"

CHROMA_COLLECTION_NAME = "akif_memory"

# ChromaDB başlatıcı
chroma_client = None
chroma_collection = None

def init_chromadb():
    global chroma_client, chroma_collection
    if chroma_client is None:
        chroma_client = chromadb.Client()
        if CHROMA_COLLECTION_NAME not in [c.name for c in chroma_client.list_collections()]:
            chroma_collection = chroma_client.create_collection(CHROMA_COLLECTION_NAME)
        else:
            chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)

def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_chat_history():
    """Sohbet geçmişini yükle"""
    try:
        with open(CHAT_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"messages": []}

def save_chat_history(history):
    """Sohbet geçmişini kaydet"""
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_memory():
    """Chat memory'yi yükle"""
    try:
        with open(CHAT_MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"user_info": {}, "conversation_insights": []}

def save_chat_memory(memory):
    """Chat memory'yi kaydet"""
    with open(CHAT_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def add_to_chat_history(user_msg, ai_msg):
    """Sohbet geçmişine yeni mesaj ekle"""
    history = load_chat_history()
    history["messages"].append({
        "user": user_msg,
        "ai": ai_msg,
        "timestamp": datetime.now().isoformat()
    })
    # Son 20 mesajı tut (çok uzun olmasın)
    if len(history["messages"]) > 20:
        history["messages"] = history["messages"][-20:]
    save_chat_history(history)

def log(msg):
    print(f"[{datetime.now().isoformat()}] {msg}")

def analyze_conversation_context(chat_history):
    """Sohbet geçmişini analiz eder ve mevcut konuyu belirler"""
    if not chat_history or not chat_history.get("messages"):
        return "Yeni konu", True
    
    messages = chat_history["messages"]
    if len(messages) < 2:
        return "Yeni konu", True
    
    # Son mesajın zamanını kontrol et
    last_msg_time = datetime.fromisoformat(messages[-1]["timestamp"])
    current_time = datetime.now()
    time_diff = current_time - last_msg_time
    
    # 5 dakikadan fazla geçtiyse yeni konu olarak kabul et
    if time_diff > timedelta(minutes=5):
        return "Yeni konu (zaman aşımı)", True
    
    # Son 3 mesajı analiz et
    recent_messages = messages[-3:]
    topics = []
    
    for msg in recent_messages:
        text = msg["user"].lower()
        if any(word in text for word in ["nasıl", "iyi", "kötü", "durum"]):
            topics.append("durum")
        elif any(word in text for word in ["ne", "yaptın", "yapıyor", "çalış"]):
            topics.append("aktivite")
        elif any(word in text for word in ["hobi", "müzik", "film", "kitap"]):
            topics.append("ilgi")
        elif any(word in text for word in ["yemek", "yemek", "aç"]):
            topics.append("yemek")
        else:
            topics.append("genel")
    
    # Konu tutarlılığını kontrol et
    if len(set(topics)) == 1:
        return f"Mevcut konu: {topics[0]}", False
    else:
        return f"Konu değişimi: {topics[-1]}", True

def get_today_info():
    """Bugünün tarihi, gün adı ve Yalova'nın hava durumu"""
    # Tarih ve gün adı
    now = datetime.now()
    tarih = now.strftime("%d %B %Y")
    gun_adi = now.strftime("%A")
    # Hava durumu
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={OWM_CITY}&appid={OWM_API_KEY}&units=metric&lang=tr"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        durum = data['weather'][0]['description'].capitalize()
        sicaklik = round(data['main']['temp'])
        hava = f"Yalova'da hava {durum}, {sicaklik}°C."
    except Exception as e:
        hava = "Hava durumu alınamadı."
    return f"Bugün {tarih}, {gun_adi}. {hava}"

def ask_ai(prompt, history, asked_questions=None, mode="chat", chat_history=None):
    system = setup_system_prompt if mode == "setup" else chat_system_prompt
    
    # Güncel tarih ve hava durumu bilgisi
    today_info = get_today_info()
    
    # Eğer setup modunda ve sorulan sorular varsa, AI'ya bildiriyoruz ki tekrar sormasın
    asked_text = ""
    if asked_questions and mode == "setup":
        asked_text = "\n\nÖNEMLİ: Aşağıdaki sorular daha önce sorulmuştur, bunları KESİNLİKLE tekrar sorma:\n" + "\n".join([f"❌ {q}" for q in asked_questions])
    
    # Chat mode'da sohbet geçmişini ekle
    chat_context = ""
    current_topic = ""
    is_new_topic = False
    user_info = ""
    personality_context = ""
    chroma_context = ""
    
    if mode == "chat" and chat_history and chat_history.get("messages"):
        chat_context = "\n\nSon 10 Mesaj:\n"
        for msg in chat_history["messages"][-10:]:  # Son 10 mesajı göster
            chat_context += f"Sen: {msg['user']}\nAkif: {msg['ai']}\n"
        
        # Konu analizi yap
        current_topic, is_new_topic = analyze_conversation_context(chat_history)
        chat_context += f"\nŞU ANKİ KONU: {current_topic}"
        if is_new_topic:
            chat_context += " (YENİ KONU - eski konuyu kapat ve yeni konuya odaklan)"
        
        # Kullanıcı bilgilerini ekle
        chat_memory = load_chat_memory()
        if chat_memory.get("user_info"):
            user_info = "\n\nKULLANICI BİLGİLERİ:\n"
            for key, value in chat_memory["user_info"].items():
                if key == "hobbies" and isinstance(value, list):
                    user_info += f"- Hobiler: {', '.join(value)}\n"
                else:
                    user_info += f"- {key.title()}: {value}\n"
        
        # Kişilik ayarlarını ekle
        settings = load_settings()
        personality_context = "\n\n" + get_personality_context(settings)
        
        # ChromaDB'den en alakalı 3 vektörü ekle
        chroma_results = query_chroma(prompt, n_results=3)
        if chroma_results:
            chroma_context = "\n\nAKİF'İN HAFIZASINDAN ÖNEMLİ ŞEYLER:\n" + "\n".join(chroma_results)
    
    # Güncel bilgi başa eklendi
    full_prompt = today_info + "\n" + prompt + asked_text + chat_context + user_info + personality_context + chroma_context + "\n\nHafıza Özeti: " + json.dumps(history, ensure_ascii=False) + "\n\nÖNEMLİ: Sadece sorulan konuya cevap ver. Gereksiz bilgi verme. Eski konuşmalara göre yanıt ver. Sormadığı şeyler hakkında açıklama yapma. Kişilik ayarlarına göre davran."
    
    payload = {
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": full_prompt}
        ]
    }
    if DEBUG_MODE:
        log(f"AI'ya gönderilen: {full_prompt}")
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json=payload, timeout=15)
    r.raise_for_status()
    resp = r.json()["choices"][0]["message"]["content"].strip()
    if DEBUG_MODE:
        log(f"AI cevabı: {resp}")
    return resp

def classify_answer(answer):
    stm, ltm, conf = {}, {}, {}
    timestamp = datetime.now().isoformat()
    
    # Anahtar üret (örneğin özet için hash veya time tabanlı id)
    key = f"info_{int(time.time())}"
    
    # Burada örnek olarak tüm cevabı STM'ye koyuyoruz,
    # İsterseniz daha gelişmiş ayırma yapabilirsiniz.
    stm[key] = {"deger": answer, "zaman": timestamp}
    return stm, ltm, conf

def parse_ai_response(response):
    """AI cevabından [accept] ile işaretlenmiş verileri çıkar"""
    stm, ltm, conf = {}, {}, {}
    timestamp = datetime.now().isoformat()
    
    # [accept] ile işaretlenmiş bölümleri bul
    import re
    accept_pattern = r'\[accept\](.*?)\[/accept\]'
    matches = re.findall(accept_pattern, response, re.DOTALL)
    
    for i, match in enumerate(matches):
        key = f"info_{int(time.time())}_{i}"
        data = match.strip()
        
        # Veri tipini belirle (daha gelişmiş sınıflandırma)
        data_lower = data.lower()
        
        # LTM (Uzun süreli bellek) - kişilik, hobi, tercihler, geçmiş deneyimler
        if any(word in data_lower for word in ["hobi", "sevdiği", "favori", "kişilik", "karakter", "tercih", "beğenir", "seviyor", "ilgi", "merak", "deneyim", "geçmiş", "çocukluk", "gençlik"]):
            ltm[key] = {"deger": data, "zaman": timestamp}
        # STM (Kısa süreli bellek) - güncel durum, şu anki hal, mevcut durum
        elif any(word in data_lower for word in ["şu an", "bugün", "güncel", "durum", "şimdi", "mevcut", "hali", "görünüyor", "hissediyor", "düşünüyor", "planlıyor"]):
            stm[key] = {"deger": data, "zaman": timestamp}
        # Confidence (Güven) - belirsiz, tahmin, muhtemelen, olabilir
        elif any(word in data_lower for word in ["muhtemelen", "olabilir", "belki", "tahmin", "sanırım", "gibi", "benzer", "yaklaşık", "ortalama"]):
            conf[key] = {"deger": data, "zaman": timestamp}
        else:
            # Varsayılan olarak LTM'ye koy (kişilik bilgileri genelde uzun süreli)
            ltm[key] = {"deger": data, "zaman": timestamp}
    
    return stm, ltm, conf, response.replace("[accept]", "").replace("[/accept]", "").strip()

def update_memory(old_mem, new_mem):
    """
    Yeni gelen belleği eskiyle günceller.
    Aynı key varsa zaman karşılaştırması yaparak güncelini tutar.
    """
    for k, v in new_mem.items():
        if k in old_mem:
            old_time = old_mem[k].get("zaman", "")
            new_time = v.get("zaman", "")
            if new_time > old_time:
                old_mem[k] = v
        else:
            old_mem[k] = v

def setup_mode():
    log("Setup modu başladı.")
    stm = load_json(STM_FILE)
    ltm = load_json(LTM_FILE)
    conf = load_json(CONFIDENCE_FILE)
    asked_data = load_json(ASKED_FILE)
    asked = set(asked_data.get("sorular", []))

    history = {"STM": stm, "LTM": ltm, "Confidence": conf}

    while True:
        # AI'dan sadece soru isteme
        if asked:
            prompt = ask_ai(
                f"Akif hakkında, aşağıdaki sorulardan FARKLI yeni ve akıllıca seçilmiş bir soru üret. Kesinlikle aynı soruları sorma:\n\nDaha önce sorulmuş sorular:\n" + "\n".join([f"- {q}" for q in asked]),
                history,
                asked_questions=asked,
                mode="setup"
            )
        else:
            prompt = ask_ai(
                "Akif hakkında ilk soru olarak akıllıca seçilmiş bir soru üret.",
                history,
                mode="setup"
            )

        # Soru kontrolü - eğer soru daha önce sorulmuşsa veya çok benzeriyse tekrar dene
        if prompt in asked:
            log("AI aynı soruyu tekrar sordu, tekrar deniyor.")
            time.sleep(1)
            continue
            
        # Benzerlik kontrolü - soru içeriğinde daha önce sorulmuş kelimeler var mı?
        prompt_lower = prompt.lower()
        is_similar = False
        for old_question in asked:
            old_lower = old_question.lower()
            # Anahtar kelimeleri kontrol et
            key_words = ["hobi", "sevdiğin", "favori", "en çok", "ne yaparsın", "nasıl", "nerede", "kim", "ne zaman"]
            for word in key_words:
                if word in prompt_lower and word in old_lower:
                    # Aynı konu hakkında soru olabilir, kontrol et
                    if any(common_word in prompt_lower and common_word in old_lower 
                           for common_word in ["hobi", "müzik", "film", "kitap", "spor", "yemek", "renk", "hayvan"]):
                        is_similar = True
                        break
            if is_similar:
                break
                
        if is_similar:
            log("AI benzer bir soru sordu, tekrar deniyor.")
            time.sleep(1)
            continue

        print(f"\nAI Soru: {prompt}")
        asked.add(prompt)

        ans = input("Senin cevabın (Dur!!! yazarsan çıkılır): ")
        if ans.strip().lower() == "dur!!!":
            break

        # AI'dan hem analiz hem de [accept] ile işaretlenmiş veri isteme
        analysis_prompt = f"Kullanıcı şu cevabı verdi: '{ans}'. Bu cevaptan üçüncü şahıs bakış açısıyla karakter analizini çıkar ve [accept] etiketleri ile işaretle. STM/LTM/Confidence ayrımını yap."
        analysis = ask_ai(analysis_prompt, history, mode="setup")
        print(f"AI Analiz: {analysis}")

        # [accept] ile işaretlenmiş verileri çıkar
        stm_new, ltm_new, conf_new, clean_analysis = parse_ai_response(analysis)
        
        # Eğer [accept] ile işaretlenmiş veri varsa göster
        if stm_new or ltm_new or conf_new:
            print(f"\nKaydedilecek veriler:")
            if stm_new:
                print(f"STM: {stm_new}")
            if ltm_new:
                print(f"LTM: {ltm_new}")
            if conf_new:
                print(f"Confidence: {conf_new}")
        else:
            print("\nKaydedilecek veri bulunamadı.")

        # Hafızayı güncelle, eski bilgiyi zamana göre değiştir
        update_memory(stm, stm_new)
        update_memory(ltm, ltm_new)
        update_memory(conf, conf_new)

        # Anlık kaydet
        save_json(STM_FILE, stm)
        save_json(LTM_FILE, ltm)
        save_json(CONFIDENCE_FILE, conf)
        save_json(ASKED_FILE, {"sorular": list(asked)})

        # Hafıza güncellenince history de güncellenmeli
        history = {"STM": stm, "LTM": ltm, "Confidence": conf}

    # Setup sonunda kaydet
    save_json(STM_FILE, stm)
    save_json(LTM_FILE, ltm)
    save_json(CONFIDENCE_FILE, conf)
    save_json(ASKED_FILE, {"sorular": list(asked)})
    open(SETUP_FLAG_FILE, "w").write("done")
    log("Setup tamamlandı.")

def chat_mode():
    log("Sohbet modu başladı.")
    stm = load_json(STM_FILE)
    ltm = load_json(LTM_FILE)
    conf = load_json(CONFIDENCE_FILE)
    chat_history = load_chat_history()
    chat_memory = load_chat_memory()

    history = {"STM": stm, "LTM": ltm, "Confidence": conf}

    print("Akif: Merhaba! Nasılsın? Ben Akif.")

    while True:
        txt = input("Sen: ")
        if txt.lower() in ["çık", "exit"]:
            break

        # Kullanıcı mesajını AI ile analiz et ve chat memory'yi güncelle
        chat_memory, insights = analyze_user_message(txt, chat_memory)
        
        # Eğer yeni bilgi çıkarıldıysa göster
        if insights:
            print(f"\n[Yeni bilgi: {', '.join(insights)}]")
            # ChromaDB'ye ekle
            for insight in insights:
                add_to_chroma(insight, metadata={"type": "insight", "timestamp": datetime.now().isoformat()})

        # Her kullanıcı mesajını ChromaDB'ye ekle
        add_to_chroma(txt, metadata={"type": "user_message", "timestamp": datetime.now().isoformat()})

        response = ask_ai(txt, history, mode="chat", chat_history=chat_history)
        print(f"Akif: {response}")

        # [accept] ile işaretlenmiş verileri çıkar
        stm_new, ltm_new, conf_new, clean_response = parse_ai_response(response)
        
        # Eğer [accept] ile işaretlenmiş veri varsa göster
        if stm_new or ltm_new or conf_new:
            print(f"\n[Kaydedilen veriler: {stm_new or ltm_new or conf_new}]")

        # Sohbet geçmişine ekle
        add_to_chat_history(txt, response)

        # Hafızayı güncelle
        update_memory(stm, stm_new)
        update_memory(ltm, ltm_new)
        update_memory(conf, conf_new)

        save_json(STM_FILE, stm)
        save_json(LTM_FILE, ltm)
        save_json(CONFIDENCE_FILE, conf)
        save_chat_memory(chat_memory)

        history = {"STM": stm, "LTM": ltm, "Confidence": conf}
        chat_history = load_chat_history()  # Güncel geçmişi yükle

def analyze_user_message(user_msg, chat_memory):
    """Kullanıcı mesajını AI ile analiz eder ve chat memory'yi günceller"""
    
    # Eğer mesaj çok kısaysa analiz etme
    if len(user_msg.strip()) < 3:
        return chat_memory, []
    
    # AI'dan kullanıcı mesajını analiz etmesini iste
    analysis_prompt = f"""
Kullanıcı şu mesajı yazdı: "{user_msg}"

Bu mesajdan kullanıcı hakkında çıkarabileceğin bilgileri analiz et. Sadece kesin ve önemli bilgileri [accept] etiketleri ile işaretle.

Örnek format:
[accept]Kullanıcının adı: Ahmet[/accept]
[accept]Kullanıcının yaşı: 25[/accept]
[accept]Kullanıcının mesleği: Öğrenci[/accept]
[accept]Kullanıcının hobisi: Müzik[/accept]
[accept]Kullanıcının ruh hali: Yorgun[/accept]

Sadece kesin bilgileri işaretle, tahmin yapma. Eğer hiçbir kesin bilgi yoksa hiçbir şey işaretleme.
"""
    
    # AI'dan analiz iste
    try:
        analysis = ask_ai(analysis_prompt, {"STM": {}, "LTM": {}, "Confidence": {}}, mode="setup")
        
        # [accept] ile işaretlenmiş verileri çıkar
        insights = []
        import re
        accept_pattern = r'\[accept\](.*?)\[/accept\]'
        matches = re.findall(accept_pattern, analysis, re.DOTALL)
        
        for match in matches:
            insight = match.strip()
            insights.append(insight)
            
            # Bilgiyi chat_memory'ye ekle
            if "adı:" in insight:
                name = insight.split("adı:")[1].strip()
                chat_memory['user_info']['name'] = name
            elif "yaşı:" in insight:
                age = insight.split("yaşı:")[1].strip()
                try:
                    chat_memory['user_info']['age'] = int(age)
                except:
                    pass
            elif "mesleği:" in insight:
                occupation = insight.split("mesleği:")[1].strip()
                chat_memory['user_info']['occupation'] = occupation
            elif "hobisi:" in insight:
                hobby = insight.split("hobisi:")[1].strip()
                if 'hobbies' not in chat_memory['user_info']:
                    chat_memory['user_info']['hobbies'] = []
                if hobby not in chat_memory['user_info']['hobbies']:
                    chat_memory['user_info']['hobbies'].append(hobby)
            elif "ruh hali:" in insight:
                mood = insight.split("ruh hali:")[1].strip()
                chat_memory['user_info']['current_mood'] = mood
        
        # Önemli bilgileri conversation_insights'a ekle
        if insights:
            timestamp = datetime.now().isoformat()
            chat_memory['conversation_insights'].append({
                'timestamp': timestamp,
                'message': user_msg,
                'insights': insights
            })
        
        return chat_memory, insights
        
    except Exception as e:
        # AI analizi başarısız olursa boş döndür
        return chat_memory, []

def load_settings():
    """Kişilik ayarlarını yükle"""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except:
        return {
            "personality": {
                "love": 30,
                "sexuality": 25,
                "emotionality": 40,
                "humor": 70,
                "boredom": 60,
                "warmth": 65
            }
        }

def get_personality_context(settings):
    """Kişilik ayarlarına göre context oluştur"""
    personality = settings.get("personality", {})
    
    context = "KİŞİLİK AYARLARI:\n"
    context += f"- Aşk/Romantik: {personality.get('love', 30)}/100 ("
    if personality.get('love', 30) < 30:
        context += "romantik konulara ilgisiz"
    elif personality.get('love', 30) < 60:
        context += "orta düzeyde ilgili"
    else:
        context += "romantik konulara ilgili"
    context += ")\n"
    
    context += f"- Cinsellik: {personality.get('sexuality', 25)}/100 ("
    if personality.get('sexuality', 25) < 30:
        context += "mesafeli"
    elif personality.get('sexuality', 25) < 60:
        context += "orta düzeyde açık"
    else:
        context += "açık"
    context += ")\n"
    
    context += f"- Duygusallık: {personality.get('emotionality', 40)}/100 ("
    if personality.get('emotionality', 40) < 30:
        context += "soğuk"
    elif personality.get('emotionality', 40) < 60:
        context += "orta düzeyde sıcak"
    else:
        context += "sıcak"
    context += ")\n"
    
    context += f"- Mizah: {personality.get('humor', 70)}/100 ("
    if personality.get('humor', 70) < 30:
        context += "ciddi"
    elif personality.get('humor', 70) < 60:
        context += "orta düzeyde esprili"
    else:
        context += "çok esprili"
    context += ")\n"
    
    context += f"- Sıkılma: {personality.get('boredom', 60)}/100 ("
    if personality.get('boredom', 60) < 30:
        context += "sabırlı"
    elif personality.get('boredom', 60) < 60:
        context += "orta düzeyde sabırsız"
    else:
        context += "çok sabırsız"
    context += ")\n"
    
    context += f"- Samimiyet: {personality.get('warmth', 65)}/100 ("
    if personality.get('warmth', 65) < 30:
        context += "soğuk"
    elif personality.get('warmth', 65) < 60:
        context += "orta düzeyde sıcak"
    else:
        context += "çok sıcak"
    context += ")\n"
    
    return context

# Vektör ekle

def add_to_chroma(text, metadata=None):
    init_chromadb()
    chroma_collection.add(
        documents=[text],
        metadatas=[metadata or {}],
        ids=[str(hash(text))]
    )

# Vektör sorgula

def query_chroma(query_text, n_results=3):
    init_chromadb()
    results = chroma_collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results.get('documents', [[]])[0] if results else []

if __name__ == "__main__":
    if not os.path.exists(SETUP_FLAG_FILE):
        setup_mode()
    else:
        log("Mod seç: [1] Sohbet | [2] Setup Eğitimi")
        c = input("Seçim: ")
        if c == "1":
            chat_mode()
        elif c == "2":
            setup_mode()
        else:
            log("Geçersiz seçim.")
