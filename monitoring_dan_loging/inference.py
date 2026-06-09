import requests
import random
import time

url = "http://localhost:8000/predict"

print("Memulai simulasi inference... Biarkan terminal ini menyala.")
for i in range(500): # Mengirim 500 request
    payload = {
        "IPK": random.uniform(2.5, 4.0),
        "Skor_Motivasi_Tertulis": random.uniform(50, 100),
        "Skor_Literasi_Finansial": random.uniform(50, 100),
        "Surat_Rekomendasi": random.randint(0, 1),
        "Aktif_Organisasi": random.randint(0, 1),
        "Jumlah_Kegiatan_Relawan": random.randint(0, 5),
        "Skor_Keaktifan_Kampus": random.uniform(0, 100),
        "Rekam_Ketidakhadiran": random.randint(0, 10),
        "Semester": 6,
        # --- DUA KOLOM TAMBAHAN ---
        "Kategori_Ekonomi": random.choice(["Rendah", "Menengah", "Tinggi"]),
        "Skor_Performa_Dinamis": random.uniform(50, 100)
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Request {i+1}: {response.json()}")
    except Exception as e:
        print("Error:", e)
        
    time.sleep(1)