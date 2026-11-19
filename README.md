# MARVIC ROV — Simulasi Autonomous Misi 5 (Payload Retrieval)
**Tugas 4 Proyek Akhir MARVIC KOMBO EDRT — Tim 1 Divisi Programming**

## 1. Ringkasan Sistem & Spesifikasi
Program `simulasi_misi5_2d.py` memvalidasi logika *autonomous* **ROV MARVIC** untuk **Misi 5** secara *real-time* (20 FPS, $\Delta t = 0.05\text{ s}$) menggunakan **Python 3 (`NumPy` & `Matplotlib`)**:

* **Lingkungan Kolam:** Ukuran $5.0\text{ m} \times 4.0\text{ m} \times 3.0\text{ m}$ dengan 4 sisi dinding (**Dinding A, B, C, D**) dan target *hook* pada kedalaman $Z = -2.0\text{ m}$.
* **Kinematika & Kontroler:** Pembaruan posisi 3D menggunakan integrasi Euler dengan gangguan arus hidrodinamika ($\sin/\cos$) serta **$P\text{-Controller}$** untuk koreksi simpangan lateral dan *depth-holding*.
* **Onboard Computer Vision (Front Webcam POV):** Proyeksi perspektif kamera depan (*pinhole*) + deteksi objek kelas tunggal `PAYLOAD` (**YOLOv8 + HSV Red Masking**) yang mengukur *Visual Servoing Error* ($\Delta x, \Delta y$ dalam piksel).
* **7 Sudut Pandang Kamera:** Tampak Samping Kiri, Samping Kanan, Atas, Depan, Belakang, 3D Isometrik ($30^\circ\text{ Kiri}, 30^\circ\text{ Atas}$), dan POV Kamera Full.

---

## 2. Cara Menjalankan Program
1. Instal pustaka yang dibutuhkan melalui terminal:
   ```bash
   pip install numpy matplotlib
   ```
2. Jalankan file simulasi:
   ```bash
   python simulasi_misi5_2d.py
   ```
   *(Gunakan `python3 simulasi_misi5_2d.py` pada terminal Linux/WSL).*
3. Pada halaman awal, pilih warna *skin* ROV jika diinginkan lalu klik **MULAI SIMULASI**. Gunakan deretan tombol di bagian bawah untuk berpindah sudut pandang, dan klik **Pilih Skenario** untuk menguji 4 sisi dinding maupun 2 skenario gagal.

---

## 3. Desain Finite State Machine (FSM) Misi 5

| Tahap FSM | Kondisi Transisi | Aksi Aktuator & Sensor |
| :--- | :--- | :--- |
| **1. IDLE** | $t_{\text{state}} \ge 1.00\text{ s}$ | Kalibrasi barometer, IMU (*Yaw*), & kamera CV di permukaan ($Z = 0\text{ m}$). |
| **2. CARI\_HOOK** | $Z \le -2.00\text{ m}$ | *Vertical thruster* menyelam ($V_z = -0.55\text{ m/s}$) sambil menahan jalur tengah. |
| **3. DEKATI\_HOOK** | Jarak ke dinding $\le 0.08\text{ m}$ | Orientasi *Yaw* menghadap dinding target, maju ($0.75\text{ m/s}$), *depth-holding* di $-2.0\text{ m}$. |
| **4. AMBIL\_PAYLOAD** | Sukses: $t \ge 1.50\text{ s}$ & CV Valid<br>Gagal: CV Loss / Capit Meleset | *Station-keeping* & mengunci *gripper*. Jika gagal, langsung beralih ke state **GAGAL**. |
| **5. LEPAS\_PAYLOAD** | $t_{\text{state}} \ge 1.20\text{ s}$ | Mundur perlahan ($-0.22\text{ m/s}$) & naik ($+0.06\text{ m/s}$) melepas *payload* dari *hook*. |
| **6. NAIK\_PERMUKAAN** | $Z \ge 0.00\text{ m}$ (**SELESAI**) | Naik diagonal ($0.72\text{ m/s}$) menuju titik tengah permukaan $(2.5, 0, 0)\text{ m}$. |

---

## 4. Evaluasi Akurasi (CTE/RMSE) & Hasil Uji 6 Skenario
* **CTE (*Cross-Track Error*):** Simpangan tegak lurus sesaat ROV terhadap garis jalur ideal:
  $$\text{CTE} = \|\vec{p} - \vec{p}_{\text{proj}}\|$$
* **RMSE (*Root Mean Square Error*):** Rata-rata akar kuadrat seluruh nilai CTE sepanjang misi:
  $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (\text{CTE}_i)^2}$$

| No | Skenario Uji | Arah Hadap (*Yaw*) | Waktu Total (Batas $600\text{ s}$) | RMSE Jalur | Status Akhir FSM |
| :-: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Dinding A (Sisi $+X$)** | $0^\circ$ | **$14.70\text{ s}$** | $0.0152\text{ m}$ | `6. SELESAI (Sukses)` |
| **2** | **Dinding B (Sisi $+Y$)** | $+90^\circ$ | **$13.10\text{ s}$** | $0.0146\text{ m}$ | `6. SELESAI (Sukses)` |
| **3** | **Dinding C (Sisi $-X$)** | $180^\circ$ | **$14.70\text{ s}$** | $0.0152\text{ m}$ | `6. SELESAI (Sukses)` |
| **4** | **Dinding D (Sisi $-Y$)** | $-90^\circ$ | **$13.10\text{ s}$** | $0.0146\text{ m}$ | `6. SELESAI (Sukses)` |
| **5** | **Gagal 1: Payload Tak Terdeteksi** | $0^\circ$ | Stop di $10.15\text{ s}$ | $0.0149\text{ m}$ | `GAGAL: CV Tak Deteksi Payload!` |
| **6** | **Gagal 2: Posisi Capit Meleset** | $0^\circ$ | Stop di $10.15\text{ s}$ | $0.0485\text{ m}$ | `GAGAL: Posisi Capit Meleset!` |

### Kesimpulan Pengujian
Seluruh skenario normal pada 4 sisi dinding (A, B, C, dan D) berhasil diselesaikan dalam waktu **$13.10\text{--}14.70\text{ detik}$** (jauh di bawah batas maksimum misi 10 menit / $600\text{ detik}$) dengan rata-rata error lintasan (**RMSE**) stabil di bawah **$2\text{ cm}$** ($0.02\text{ m}$), serta kedua skenario *fail-safe* berhasil terdeteksi secara otomatis oleh *state machine*.