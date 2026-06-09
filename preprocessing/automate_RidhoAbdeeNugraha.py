import pandas as pd
import numpy as np
import os

def terapkan_logika_bisnis(df_input):
    df_temp = df_input.copy()
    
    # Binning pendapatan orang tua
    batas_pendapatan = [0, 5000000, np.inf]
    label_pendapatan = ['Rentan', 'Stabil']
    df_temp['Kategori_Ekonomi'] = pd.cut(df_temp['Pendapatan_Orang_Tua'], bins=batas_pendapatan, labels=label_pendapatan)
    
    # Perhitungan skor performa dinamis
    skor_ipk = (df_temp['IPK'] / 4.0) * 100
    skor_keaktifan = df_temp['Skor_Keaktifan_Kampus']
    skor_relawan = df_temp['Jumlah_Kegiatan_Relawan'] * 10
    df_temp['Skor_Performa_Dinamis'] = (skor_ipk * 0.60) + (skor_keaktifan * 0.20) + (skor_relawan * 0.20)
    
    # Menghapus kolom yang tidak dibutuhkan model
    df_temp = df_temp.drop(columns=['Pendapatan_Orang_Tua'])
    return df_temp

def main():
    # 1. Mendefinisikan Path (sesuai struktur folder)
    # File ini dijalankan di root directory repository, sehingga path-nya relatif dari root
    input_path = 'dataset-kelayakan-beasiswa_raw/dataset-kelayakan-beasiswa_raw.csv'
    output_dir = 'preprocessing/dataset-kelayakan-beasiswa_preprocessing'
    output_path = f'{output_dir}/dataset_kelayakan_beasiswa_clean.csv'

    print("=== Memulai Automasi Preprocessing ===")
    
    # 2. Load Data
    print("1. Membaca dataset raw...")
    df = pd.read_csv(input_path)

    # 3. Feature Selection
    top_features = [
        'Pendapatan_Orang_Tua', 'IPK', 'Skor_Motivasi_Tertulis',
        'Skor_Literasi_Finansial', 'Surat_Rekomendasi', 'Aktif_Organisasi',
        'Jumlah_Kegiatan_Relawan', 'Skor_Keaktifan_Kampus', 'Rekam_Ketidakhadiran',
        'Semester', 'Status_Kelayakan'
    ]
    df_selected = df[top_features].copy()

    # 4. Feature Engineering
    print("2. Menerapkan logika bisnis (Feature Engineering)...")
    df_processed = terapkan_logika_bisnis(df_selected)

    # 5. Encoding Target
    print("3. Melakukan encoding target...")
    df_processed['Status_Kelayakan'] = df_processed['Status_Kelayakan'].map({'Layak': 1, 'Tidak Layak': 0})

    # 6. Menyimpan Data Bersih
    # Memastikan folder output tersedia (jika belum ada, buat otomatis)
    os.makedirs(output_dir, exist_ok=True)
    
    df_processed.to_csv(output_path, index=False)
    print(f"4. Selesai! Dataset bersih berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    main()