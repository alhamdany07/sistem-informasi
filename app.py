import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ==========================================
# 1. LOAD DATASET DARI FILE CSV
# ==========================================
try:
    df = pd.read_csv('dataset_buku.csv')
    if 'rating' not in df.columns:
        df['rating'] = 4.5
    if 'sinopsis' not in df.columns:
        df['sinopsis'] = 'Sinopsis untuk buku ini sedang dalam proses pembaruan oleh tim pustakawan.'
except Exception as e:
    print(f"Error membaca CSV: {e}")
    df = pd.DataFrame()

# Engine 1: Content-Based Filtering (Berdasarkan Genre)
if not df.empty:
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(df['genre'])
    similarity_score = cosine_similarity(cv_matrix)
else:
    similarity_score = None

# Data Ulasan Awal untuk Dashboard Review Canvas
ulasan_dummy = [
    {"user": "Andi Saputra", "skor": 5, "komentar": "Alur ceritanya sangat menyentuh hati, penokohannya luar biasa!"},
    {"user": "Siti Rahma", "skor": 4, "komentar": "Bahasanya puitis dan dalam. Sangat merekomendasikan buku ini."}
]

def ambil_rekomendasi(judul_buku, k=4):
    if similarity_score is None or df.empty:
        return []
    try:
        idx = df[df['judul'].str.lower() == judul_buku.lower()].index[0]
        skor_kemiripan = list(enumerate(similarity_score[idx]))
        skor_kemiripan = sorted(skor_kemiripan, key=lambda x: x[1], reverse=True)
        buku_mirip_idx = [i[0] for i in skor_kemiripan[1:k+1]]
        return df.iloc[buku_mirip_idx].to_dict(orient='records')
    except:
        return []

# ==========================================
# 2. ROUTING / CONTROLLER FLASK
# ==========================================
@app.route('/')
def index():
    if df.empty:
        return render_template('index.html', buku=[], populer=[])
    
    # Ambil semua buku untuk katalog utama
    daftar_buku = df.to_dict(orient='records')
    
    # 🌟 FITUR BARU: Engine 2 - Popularity-Based Filtering (Urutkan dari rating tertinggi)
    buku_populer = df.sort_values(by='rating', ascending=False).head(4).to_dict(orient='records')
    
    return render_template('index.html', buku=daftar_buku, populer=buku_populer)

@app.route('/rekomendasi/<judul>')
def rekomendasi(judul):
    buku_terpilih = df[df['judul'].str.lower() == judul.lower()]
    if buku_terpilih.empty:
        return render_template('rekomendasi.html', judul_dicari=judul, hasil=[], error=True)

    hasil_rekomendasi = ambil_rekomendasi(judul, k=4)
    info_buku = buku_terpilih.to_dict(orient='records')[0]
    
    # Bobot Canvas Dinamis Per Buku
    buku_id = int(info_buku.get('id', 1)) # Pakai .get() jaga-jaga kalau ID kosong
    info_buku['plot_weight'] = 80 + (buku_id * 4) % 19
    info_buku['diksi_weight'] = 75 + (buku_id * 7) % 21

    return render_template('rekomendasi.html', judul_dicari=info_buku['judul'], info=info_buku, hasil=hasil_rekomendasi, ulasan=ulasan_dummy, error=False)

if __name__ == '__main__':
    # Modifikasi khusus untuk Render.com agar membaca PORT secara dinamis
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)