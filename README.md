# Benchmark Komparasi 3 Model (1 ML vs 2 Deep Learning)
### Multi-Class News Classification pada Dataset AG News (120.000 Sampel)

Proyek ini mengevaluasi secara empiris performa klasifikasi, durasi komputasi training, dan efisiensi memori hardware antara 3 generasi pemodelan:
1. **Model 1 (Machine Learning Klasik)**: TF-IDF (10.000 N-Gram features) + `LogisticRegression` (L-BFGS)
2. **Model 2 (Deep Learning Rekuren)**: Custom Bidirectional LSTM (`BiLSTM`) 2-layer dengan Embedding Layer pada PyTorch CUDA
3. **Model 3 (Deep Learning Transformer)**: Pretrained `DistilBERT` (`distilbert-base-uncased`) Fine-Tuning dengan Automatic Mixed Precision (`fp16`)

---

## Spesifikasi Lingkungan Komputasi
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Deep Learning Framework**: PyTorch with CUDA acceleration (`torch 2.7.0+cu128`)
- **NLP Library**: Hugging Face `transformers` & `datasets`
- **Safeguard Anti-OOM**: Automatic Mixed Precision (FP16), batch size 16 untuk Transformer, serta inter-model cache clearance (`torch.cuda.empty_cache()` dan `gc.collect()`)
- **Total Estimasi Durasi Training**: ~30 - 40 Menit

---

## Struktur Direktori
- `dataset/`: Subfolder penyimpanan dataset lokal
  - `train.csv`: 120.000 baris data latih
  - `test.csv`: 7.600 baris data uji
- `benchmark_3models_nlp.ipynb`: Jupyter Notebook utama berisi pipeline lengkap (ingestion, EDA, training, benchmarking, dan visualisasi)
- `requirements.txt`: Daftar dependensi Python

---

## Panduan Menjalankan Notebook
1. Buka Jupyter Notebook atau IDE yang mendukung notebook:
   ```bash
   jupyter notebook benchmark_3models_nlp.ipynb
   ```
2. Jalankan cell secara berurutan.
3. Seluruh dataset akan dibaca secara otomatis dari folder lokal `./dataset/`.
