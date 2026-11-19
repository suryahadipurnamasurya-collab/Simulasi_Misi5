import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ==========================================
# KONFIGURASI PARAMETER FISIKA & MISI 5 ROV
# ==========================================
DT = 0.05        # Time step (20 FPS real-time)
POOL_LEN = 5.0   # Panjang kolam sumbu X: 0.0 s/d 5.0 m
POOL_WID = 4.0   # Lebar kolam sumbu Y: -2.0 s/d +2.0 m
POOL_DEP = 3.0   # Kedalaman kolam sumbu Z: 0.0 s/d -3.0 m

SKINS = [
    ('Kuning', '#fbc02d'),
    ('Biru',   '#29b6f6'),
    ('Hijau',  '#00e676'),
    ('Merah',  '#ef5350'),
]

# 6 Skenario Wajib Sesuai PDF Tugas 4 (4 Sisi Dinding A, B, C, D + 2 Skenario Gagal)
DAFTAR_SKENARIO = [
    'Dinding A (Sisi +X | Yaw 0°)',
    'Dinding B (Sisi +Y | Yaw +90°)',
    'Dinding C (Sisi -X | Yaw 180°)',
    'Dinding D (Sisi -Y | Yaw -90°)',
    'Gagal: Payload Tak Terdeteksi',
    'Gagal: Posisi Capit Meleset'
]

# Konfigurasi Koordinat Waypoint & Hook untuk Tiap Sisi Dinding (A, B, C, D)
KONFIG_DINDING = {
    'A': {
        'start': np.array([1.2, 0.0, 0.0]),
        'dive':  np.array([1.2, 0.0, -2.0]),
        'approach': np.array([4.20, 0.0, -2.0]),
        'hook':  np.array([4.45, 0.0, -2.0]),
        'surface': np.array([2.50, 0.0, 0.0]),
        'yaw_deg': 0.0,
        'normal_vec': np.array([1.0, 0.0, 0.0])
    },
    'B': {
        'start': np.array([2.50, -0.5, 0.0]),
        'dive':  np.array([2.50, -0.5, -2.0]),
        'approach': np.array([2.50, 1.50, -2.0]),
        'hook':  np.array([2.50, 1.75, -2.0]),
        'surface': np.array([2.50, 0.0, 0.0]),
        'yaw_deg': 90.0,
        'normal_vec': np.array([0.0, 1.0, 0.0])
    },
    'C': {
        'start': np.array([3.8, 0.0, 0.0]),
        'dive':  np.array([3.8, 0.0, -2.0]),
        'approach': np.array([0.80, 0.0, -2.0]),
        'hook':  np.array([0.55, 0.0, -2.0]),
        'surface': np.array([2.50, 0.0, 0.0]),
        'yaw_deg': 180.0,
        'normal_vec': np.array([-1.0, 0.0, 0.0])
    },
    'D': {
        'start': np.array([2.50, 0.5, 0.0]),
        'dive':  np.array([2.50, 0.5, -2.0]),
        'approach': np.array([2.50, -1.50, -2.0]),
        'hook':  np.array([2.50, -1.75, -2.0]),
        'surface': np.array([2.50, 0.0, 0.0]),
        'yaw_deg': -90.0,
        'normal_vec': np.array([0.0, -1.0, 0.0])
    }
}

class ROVSimulator2D:
    def __init__(self):
        self.screen = 'MENU'
        self.skenario = DAFTAR_SKENARIO[0]
        self.view_mode = 'SAMPING_KIRI'
        self.skin_idx = 0
        self.paused = False
        self.dropdown_open = False
        np.random.seed(42)
        self.particles_u = np.random.uniform(-1.2, 1.2, 18)
        self.particles_v = np.random.uniform(-0.85, 0.85, 18)
        self.reset()

    @property
    def skin_name(self):
        return SKINS[self.skin_idx][0]

    @property
    def skin_color(self):
        return SKINS[self.skin_idx][1]

    def next_skin(self):
        self.skin_idx = (self.skin_idx + 1) % len(SKINS)

    def get_sisi_dinding(self):
        if 'Dinding B' in self.skenario:
            return 'B'
        elif 'Dinding C' in self.skenario:
            return 'C'
        elif 'Dinding D' in self.skenario:
            return 'D'
        return 'A'

    def reset(self):
        self.sisi = self.get_sisi_dinding()
        cfg = KONFIG_DINDING[self.sisi]
        self.wp_start = cfg['start'].copy()
        self.wp_dive = cfg['dive'].copy()
        self.wp_approach = cfg['approach'].copy()
        self.hook_pos = cfg['hook'].copy()
        self.wp_surface = cfg['surface'].copy()
        self.yaw_deg = cfg['yaw_deg']
        self.forward_vec = cfg['normal_vec'].copy()

        self.pos = self.wp_start.copy()
        self.vel = np.array([0.0, 0.0, 0.0])
        self.payload_pos = self.hook_pos.copy()
        self.payload_attached = False

        self.state = '1. IDLE'
        self.state_idx = 0
        self.elapsed_time = 0.0
        self.state_timer = 0.0
        self.state_log = ['[00.0s] STATE -> 1. IDLE (Siap)']

        self.traj_x = [float(self.pos[0])]
        self.traj_y = [float(self.pos[1])]
        self.traj_z = [float(self.pos[2])]
        self.time_hist = [0.0]
        self.depth_hist = [float(self.pos[2])]
        self.dist_hist = [0.0]
        self.cte_hist = [0.0]
        self.rmse_hist = [0.0]
        self.sq_error_sum = 0.0
        self.sample_count = 1

    def catat_log(self, pesan_state):
        waktu_str = f"[{self.elapsed_time:04.1f}s] {pesan_state}"
        self.state_log.append(waktu_str)
        if len(self.state_log) > 4:
            self.state_log.pop(0)

    def hitung_cte_segmen(self, p, a, b):
        ab = b - a
        ap = p - a
        ab_norm_sq = np.dot(ab, ab)
        if ab_norm_sq == 0:
            return float(np.linalg.norm(ap))
        t = np.clip(np.dot(ap, ab) / ab_norm_sq, 0.0, 1.0)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def step(self):
        if self.paused or self.screen != 'SIM':
            return

        if '6. SELESAI' in self.state or 'GAGAL' in self.state:
            self.state_timer += DT
            if 'Posisi Capit Meleset' in self.skenario and self.payload_pos[2] > -2.85:
                self.payload_pos[2] -= 1.1 * DT
            self.vel = np.array([0.0, 0.0, 0.0])
            if self.state_timer >= 3.8:
                self.reset()
            return

        self.elapsed_time += DT
        self.state_timer += DT

        # Gangguan arus hidrodinamika bawah air
        disturb_1 = 0.05 * np.sin(2.5 * self.elapsed_time)
        disturb_z = 0.04 * np.cos(2.0 * self.elapsed_time)
        lat_vec = np.array([-self.forward_vec[1], self.forward_vec[0], 0.0])

        if self.state == '1. IDLE':
            self.vel = np.array([0.0, 0.0, 0.0])
            cte = float(np.linalg.norm(self.pos - self.wp_start))
            if self.state_timer >= 1.0:
                self.state = f'2. CARI_HOOK (Dinding {self.sisi})'
                self.state_idx = 1
                self.state_timer = 0.0
                self.catat_log(f'-> 2. CARI_HOOK (Selam Z=-2m)')

        elif '2. CARI_HOOK' in self.state:
            err_lat = np.dot(self.pos - self.wp_start, lat_vec)
            v_lat = (-1.6 * err_lat + disturb_1) * lat_vec
            self.vel = np.array([v_lat[0], v_lat[1], -0.55])
            self.pos += self.vel * DT
            cte = self.hitung_cte_segmen(self.pos, self.wp_start, self.wp_dive)

            if self.pos[2] <= -2.0:
                self.pos[2] = -2.0
                self.state = f'3. DEKATI_HOOK (Dinding {self.sisi})'
                self.state_idx = 2
                self.state_timer = 0.0
                self.catat_log(f'-> 3. DEKATI_HOOK (Yaw {self.yaw_deg:.0f}°)')

        elif '3. DEKATI_HOOK' in self.state:
            arah = self.wp_approach - self.pos
            jarak = float(np.linalg.norm(arah[:2]))
            err_lat = np.dot(self.pos - self.wp_dive, lat_vec)
            v_lat = (-1.8 * err_lat + disturb_1) * lat_vec
            vz_cmd = -2.0 * (self.pos[2] - (-2.0)) + disturb_z

            self.vel = self.forward_vec * 0.75 + v_lat + np.array([0.0, 0.0, vz_cmd])
            self.pos += self.vel * DT
            cte = self.hitung_cte_segmen(self.pos, self.wp_dive, self.wp_approach)

            if jarak <= 0.08 or np.dot(self.wp_approach - self.pos, self.forward_vec) <= 0.0:
                self.pos[0], self.pos[1] = self.wp_approach[0], self.wp_approach[1]
                self.state = '4. AMBIL_PAYLOAD'
                self.state_idx = 3
                self.state_timer = 0.0
                self.catat_log('-> 4. AMBIL_PAYLOAD (CV Lock)')

        elif self.state == '4. AMBIL_PAYLOAD':
            # Jika skenario gagal posisi meleset, tambahkan offset lateral pada ROV
            if 'Posisi Capit Meleset' in self.skenario:
                self.vel = lat_vec * 0.18 + np.array([0.0, 0.0, disturb_z * 0.2])
            else:
                self.vel = lat_vec * (disturb_1 * 0.2) + np.array([0.0, 0.0, disturb_z * 0.2])
            self.pos += self.vel * DT
            cte = float(np.linalg.norm(self.pos - self.wp_approach))

            if self.state_timer >= 1.5:
                if 'Payload Tak Terdeteksi' in self.skenario:
                    self.state = 'GAGAL: CV Tak Deteksi Payload!'
                    self.state_timer = 0.0
                    self.catat_log('-> GAGAL: CV Loss / No Object!')
                elif 'Posisi Capit Meleset' in self.skenario:
                    self.state = 'GAGAL: Posisi Capit Meleset!'
                    self.state_timer = 0.0
                    self.catat_log('-> GAGAL: Gripper Offset/Jatuh!')
                else:
                    self.payload_attached = True
                    self.state = '5. LEPAS_PAYLOAD'
                    self.state_idx = 4
                    self.state_timer = 0.0
                    self.catat_log('-> 5. LEPAS_PAYLOAD (Unhook)')

        elif self.state == '5. LEPAS_PAYLOAD':
            self.vel = -self.forward_vec * 0.22 + np.array([0.0, 0.0, 0.06])
            self.pos += self.vel * DT
            if self.payload_attached:
                self.payload_pos = self.pos + self.forward_vec * 0.25
            cte = float(np.linalg.norm(self.pos - self.wp_approach))

            if self.state_timer >= 1.2:
                self.state = '6. NAIK_PERMUKAAN'
                self.state_idx = 5
                self.state_timer = 0.0
                self.catat_log('-> 6. NAIK_PERMUKAAN (Ascent)')

        elif self.state == '6. NAIK_PERMUKAAN':
            arah = self.wp_surface - self.pos
            jarak = float(np.linalg.norm(arah))
            if jarak > 0.08:
                v_ideal = (arah / jarak) * 0.72
                self.vel = v_ideal + lat_vec * (disturb_1 * 0.3)
                self.pos += self.vel * DT
            cte = self.hitung_cte_segmen(self.pos, self.wp_approach, self.wp_surface)

            if self.payload_attached:
                self.payload_pos = self.pos + self.forward_vec * 0.25

            if self.pos[2] >= 0.0 or jarak <= 0.08:
                self.pos[2] = 0.0
                self.vel = np.array([0.0, 0.0, 0.0])
                self.state = f'6. SELESAI (Dinding {self.sisi} Sukses)'
                self.state_idx = 6
                self.state_timer = 0.0
                self.catat_log(f'-> SELESAI ({self.elapsed_time:.1f}s < 600s)')

        self.particles_u = ((self.particles_u - np.dot(self.vel, lat_vec) * 0.35 + 1.2) % 2.4) - 1.2
        self.particles_v = ((self.particles_v - self.vel[2] * 0.35 + 0.85) % 1.7) - 0.85

        self.traj_x.append(float(self.pos[0]))
        self.traj_y.append(float(self.pos[1]))
        self.traj_z.append(float(self.pos[2]))
        self.time_hist.append(float(self.elapsed_time))
        self.depth_hist.append(float(self.pos[2]))
        self.dist_hist.append(float(np.linalg.norm(self.pos[:2] - self.wp_start[:2])))

        self.sample_count += 1
        self.sq_error_sum += cte ** 2
        rmse = float(np.sqrt(self.sq_error_sum / self.sample_count))

        self.cte_hist.append(cte)
        self.rmse_hist.append(rmse)


# ==========================================
# SETUP FIGURE & GRID LAYOUT
# ==========================================
sim = ROVSimulator2D()

plt.style.use('dark_background')
fig = plt.figure(figsize=(15.2, 8.6))
fig.canvas.manager.set_window_title('MARVIC ROV - project akhir tim 1')

gs = fig.add_gridspec(2, 3, height_ratios=[3.15, 2.2], width_ratios=[1.15, 1.15, 1.12],
                      hspace=0.36, wspace=0.23, left=0.05, right=0.97, top=0.90, bottom=0.13)

ax_main = fig.add_subplot(gs[0, 0:2])                   # Layar Utama 2D
ax_3d = fig.add_subplot(gs[0, 0:2], projection='3d')    # Layar Utama 3D
ax_cam = fig.add_subplot(gs[0, 2])                      # Panel Webcam CV POV Depan

ax_pos = fig.add_subplot(gs[1, 0])                      # Grafik Posisi & Kedalaman terhadap Waktu (Syarat PDF No.5)
ax_err = fig.add_subplot(gs[1, 1])                      # Grafik CTE & RMSE Real-Time
ax_hud = fig.add_subplot(gs[1, 2])                      # Panel Telemetri & Log State FSM

sim_title = fig.suptitle('SIMULASI AUTONOMOUS ROV MARVIC - MISI 5 (UJI 4 DINDING A/B/C/D, FAIL-SAFE & COMPUTER VISION)',
                         fontsize=12.0, fontweight='bold', color='#00e5ff', y=0.96)

# --- AXES HALAMAN AWAL (MENU) ---
ax_menu = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax_menu.set_facecolor('#0b1116')

def gambar_menu():
    ax_menu.clear()
    ax_menu.set_facecolor('#0b1116')
    ax_menu.set_xlim(0, 10)
    ax_menu.set_ylim(0, 10)
    ax_menu.axis('off')

    ax_menu.add_patch(patches.FancyBboxPatch((0.6, 0.6), 8.8, 8.8, boxstyle="round,pad=0.2",
                                             facecolor='#101a22', edgecolor='#00e5ff', linewidth=2))

    ax_menu.text(5.0, 8.45, 'SELAMAT DATANG DI PROJECT AKHIR TIM KAMI',
                 ha='center', va='center', fontsize=20, fontweight='bold', color='#00e5ff')
    ax_menu.text(5.0, 7.80, '( DIVISI PROGRAMMING - MARVIC ROV TIM 1 )',
                 ha='center', va='center', fontsize=13, fontweight='bold', color='#ffd54f')
    ax_menu.text(5.0, 7.25, 'Tugas 4: Simulasi Autonomous Misi 5 (Uji 4 Sisi Dinding A, B, C, D + 2 Skenario Gagal + CV POV)',
                 ha='center', va='center', fontsize=10, color='#b0bec5')

    ax_menu.add_patch(patches.FancyBboxPatch((3.3, 4.45), 3.4, 2.2, boxstyle="round,pad=0.15",
                                             facecolor='#0d2533', edgecolor='#26c6da', linewidth=1.3))
    cx, cy = 5.0, 5.55
    ax_menu.add_patch(patches.Rectangle((cx - 0.75, cy - 0.38), 1.5, 0.76,
                                        facecolor=sim.skin_color, edgecolor='white', linewidth=2))
    ax_menu.add_patch(patches.Rectangle((cx - 0.60, cy + 0.25), 1.2, 0.18, facecolor='#ff7043'))
    ax_menu.add_patch(patches.Rectangle((cx - 0.20, cy + 0.43), 0.4, 0.12, facecolor='#29b6f6'))
    ax_menu.add_patch(patches.Wedge((cx + 0.75, cy), 0.22, -90, 90,
                                    facecolor='#263238', edgecolor='#00e5ff', linewidth=2))
    ax_menu.plot([cx + 0.75, cx + 1.15], [cy + 0.12, cy + 0.22], color='#e0e0e0', linewidth=3.5)
    ax_menu.plot([cx + 0.75, cx + 1.15], [cy - 0.12, cy - 0.22], color='#e0e0e0', linewidth=3.5)

    ax_menu.text(5.0, 4.65, f'Skin Kapal Aktif: {sim.skin_name.upper()}',
                 ha='center', va='center', fontsize=10, fontweight='bold', color=sim.skin_color)

    fig.canvas.draw_idle()


# ==========================================
# FUNGSI WEBCAM POV + COMPUTER VISION KHUSUS DETEKSI PAYLOAD
# ==========================================
def gambar_webcam_pov(ax, is_main=False):
    ax.clear()
    ax.set_facecolor('#041b24')
    judul_cam = f'LAYAR UTAMA: POV CV MARVIC (DINDING {sim.sisi})' if is_main else f'POV WEBCAM + CV (TARGET: DINDING {sim.sisi})'
    ax.set_title(judul_cam, fontsize=9.2 if not is_main else 11, fontweight='bold', color='#00e5ff')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.92, 0.92)
    ax.set_xticks([])
    ax.set_yticks([])

    x, y, z = sim.pos
    px, py, pz = sim.payload_pos
    hx, hy, hz = sim.hook_pos
    focal = 1.15

    # Koordinat lokal kamera (sumbu pandang mengikuti forward_vec menuju dinding aktif A/B/C/D)
    fwd = sim.forward_vec
    right_vec = np.array([fwd[1], -fwd[0], 0.0])

    rel_hook = sim.hook_pos - (sim.pos + fwd * 0.20)
    dist_hook = max(0.18, float(np.dot(rel_hook, fwd)))
    dist_wall = dist_hook + 0.12

    surf_v = focal * (0.0 - z) / dist_wall
    floor_v = focal * (-POOL_DEP - z) / dist_wall
    if surf_v < 0.95:
        ax.axhspan(max(-0.92, surf_v), 0.92, color='#4dd0e1', alpha=0.20)
    if floor_v > -0.95:
        ax.axhspan(-0.92, min(0.92, floor_v), color='#020d12', alpha=0.65)

    wall_u_left = -focal * 1.8 / dist_wall
    wall_u_right = focal * 1.8 / dist_wall
    wall_v_bot = focal * (-POOL_DEP - z) / dist_wall
    wall_v_top = focal * (0.0 - z) / dist_wall
    ax.add_patch(patches.Rectangle((wall_u_left, wall_v_bot), wall_u_right - wall_u_left, wall_v_top - wall_v_bot,
                                   facecolor='#37474f', edgecolor='#78909c', alpha=0.55, linewidth=1.0))

    # Label huruf Dinding (A / B / C / D) di dinding kolam depan kamera
    ax.text(0.0, min(0.65, wall_v_top - 0.18), f'DINDING {sim.sisi}',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#cfd8dc', alpha=0.25)

    hook_u = focal * float(np.dot(rel_hook, right_vec)) / dist_hook
    hook_v = focal * (hz - z) / dist_hook
    hook_r = focal * 0.05 / dist_hook
    ax.plot([hook_u, hook_u], [hook_v, hook_v + hook_r * 2.2], color='#9e9e9e', linewidth=2.5, alpha=0.7)
    ax.add_patch(patches.Circle((hook_u, hook_v), hook_r, facecolor='#757575', edgecolor='#bdbdbd', alpha=0.7))

    # Proyeksi Bola Payload Merah
    rel_pay = sim.payload_pos - (sim.pos + fwd * 0.20)
    dist_pay = max(0.16, float(np.dot(rel_pay, fwd)))
    if sim.payload_attached:
        pay_u, pay_v, pay_r = 0.0, -0.32, 0.27
    else:
        pay_u = focal * float(np.dot(rel_pay, right_vec)) / dist_pay
        pay_v = focal * (pz - z) / dist_pay
        pay_r = float(np.clip(focal * 0.12 / dist_pay, 0.035, 0.40))

    ax.add_patch(patches.Circle((pay_u, pay_v), pay_r, facecolor='#ff1744', edgecolor='white', linewidth=1.2, zorder=5))
    ax.add_patch(patches.Circle((pay_u - pay_r * 0.28, pay_v + pay_r * 0.28), pay_r * 0.20,
                                facecolor='#ff8a80', alpha=0.85, zorder=6))

    # Fitur Computer Vision (Deteksi Payload & Skenario Gagal Tak Terdeteksi)
    cv_blind = ('Payload Tak Terdeteksi' in sim.skenario) and (sim.state_idx >= 2 or 'GAGAL' in sim.state)
    payload_in_fov = (-1.18 < pay_u < 1.18) and (-0.86 < pay_v < 0.86) and (not cv_blind)
    jarak_3d = float(np.linalg.norm(sim.payload_pos - sim.pos))

    if payload_in_fov:
        conf_score = float(np.clip(0.99 - 0.035 * jarak_3d, 0.84, 0.99))
        err_px_x = int(pay_u * 320)
        err_px_y = int(pay_v * 240)
        area_px = int((pay_r * 240) ** 2 * np.pi)

        is_locked = (dist_pay < 0.85 or sim.payload_attached) and ('GAGAL' not in sim.state)
        cv_color = '#00e676' if is_locked else '#ffea00'

        ax.add_patch(patches.Circle((pay_u, pay_v), pay_r * 1.06, fill=False,
                                    edgecolor='#00e5ff', linestyle=':', linewidth=1.5, zorder=7))
        ax.plot([pay_u - 0.04, pay_u + 0.04], [pay_v, pay_v], color=cv_color, linewidth=1.6, zorder=8)
        ax.plot([pay_u, pay_u], [pay_v - 0.04, pay_v + 0.04], color=cv_color, linewidth=1.6, zorder=8)
        ax.plot([0.0, pay_u], [0.0, pay_v], linestyle='--', color='#00e5ff', linewidth=1.2, alpha=0.85, zorder=7)

        pad = pay_r * 1.38
        x0, y0, x1, y1 = pay_u - pad, pay_v - pad, pay_u + pad, pay_v + pad
        ax.add_patch(patches.Rectangle((x0, y0), pad * 2, pad * 2,
                                       fill=False, edgecolor=cv_color, linestyle='--', linewidth=1.0, alpha=0.7, zorder=7))
        clen = pad * 0.42
        for cx_c, cy_c, sx, sy in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
            ax.plot([cx_c, cx_c + sx * clen], [cy_c, cy_c], color=cv_color, linewidth=2.4, zorder=8)
            ax.plot([cx_c, cx_c], [cy_c, cy_c + sy * clen], color=cv_color, linewidth=2.4, zorder=8)

        label_cv = f"class: PAYLOAD [{conf_score*100:.0f}%]\ndist:{jarak_3d:.2f}m | err:({err_px_x:+d},{err_px_y:+d})px"
        ax.text(pay_u, min(0.74, y1 + 0.04), label_cv,
                ha='center', va='bottom', fontsize=6.2 if not is_main else 8.0,
                family='monospace', fontweight='bold', color=cv_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#000000', alpha=0.78, edgecolor=cv_color), zorder=9)
    else:
        area_px = 0
        if cv_blind:
            ax.text(0.0, 0.25, '⚠ CV ERROR: PAYLOAD NOT DETECTED\n(CONFIDENCE < THRESHOLD 0.50)',
                    ha='center', va='center', fontsize=7.0 if not is_main else 9.0,
                    family='monospace', fontweight='bold', color='#ff1744',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000', edgecolor='#ff1744'), zorder=9)

    # PIP Jendela Binary Masking (HSV Red Mask)
    m_x0, m_y0, m_w, m_h = 0.62, -0.86, 0.58, 0.44
    ax.add_patch(patches.Rectangle((m_x0, m_y0), m_w, m_h,
                                   facecolor='#000000', edgecolor='#00e676', linewidth=1.1, zorder=10))
    ax.text(m_x0 + m_w / 2, m_y0 + m_h - 0.06, 'CV HSV RED MASK',
            ha='center', va='top', fontsize=5.3 if not is_main else 6.8,
            family='monospace', fontweight='bold', color='#00e676', zorder=11)

    if payload_in_fov:
        mask_u = m_x0 + m_w / 2 + (pay_u / 1.25) * (m_w * 0.38)
        mask_v = m_y0 + m_h / 2 - 0.03 + (pay_v / 0.92) * (m_h * 0.35)
        mask_r = np.clip(pay_r * 0.22, 0.02, 0.11)
        ax.add_patch(patches.Circle((mask_u, mask_v), mask_r,
                                    facecolor='white', edgecolor='#00e676', linewidth=0.8, zorder=11))

    ax.scatter(sim.particles_u, sim.particles_v, s=5, color='#b2ebf2', alpha=0.30, zorder=4)

    grip_offset = 0.14 if sim.payload_attached or sim.state_idx in [3, 4] else 0.28
    ax.plot([-0.85, -grip_offset - 0.12, -grip_offset], [-0.92, -0.58, -0.40], color='#cfd8dc', linewidth=4.0, zorder=9)
    ax.plot([0.55, grip_offset + 0.10, grip_offset], [-0.92, -0.58, -0.40], color='#cfd8dc', linewidth=4.0, zorder=9)

    ax.plot([-0.12, 0.12], [0.0, 0.0], color='#00e5ff', linewidth=0.9, alpha=0.75, zorder=10)
    ax.plot([0.0, 0.0], [-0.12, 0.12], color='#00e5ff', linewidth=0.9, alpha=0.75, zorder=10)

    rec_color = '#00e676' if payload_in_fov else '#ff1744'
    status_det = 'DETECTED (1)' if payload_in_fov else 'NO TARGET (0)'
    ax.text(-1.19, 0.81, f'● CV: YOLOv8-Payload | {status_det}',
            fontsize=6.2 if not is_main else 8.0, fontweight='bold', family='monospace', color=rec_color, zorder=10)
    ax.text(1.19, 0.81, f'Z:{abs(z):.2f}m', ha='right',
            fontsize=6.5 if not is_main else 8.0, fontweight='bold', family='monospace', color='#00e5ff', zorder=10)
    ax.text(-1.19, -0.84, f'YAW:{sim.yaw_deg:+.0f}° | AREA:{area_px}px²',
            fontsize=6.0 if not is_main else 7.6, family='monospace', color='#80deea', zorder=10)

    ax.add_patch(patches.Rectangle((-1.23, -0.90), 2.46, 1.80, fill=False,
                                   edgecolor='#00e5ff', linewidth=1.2, alpha=0.6, zorder=11))


# ==========================================
# FUNGSI GAMBAR SIMULASI EKSTERNAL (2D & 3D)
# ==========================================
def gambar_layar_2d(ax, mode):
    ax.clear()
    ax.grid(True, linestyle='--', alpha=0.28, color='#888888')

    x, y, z = sim.pos
    px, py, pz = sim.payload_pos
    hx, hy, hz = sim.hook_pos
    warna_rov = sim.skin_color

    if mode in ['SAMPING_KIRI', 'SAMPING_KANAN']:
        judul = f'TAMPAK SAMPING KIRI (X - Z) | TARGET: DINDING {sim.sisi}' if mode == 'SAMPING_KIRI' else f'TAMPAK SAMPING KANAN (X - Z) | TARGET: DINDING {sim.sisi}'
        ax.set_title(judul, fontsize=10.0, fontweight='bold', color='#ffd54f')
        ax.set_xlabel('Posisi Horizontal X (m)', fontsize=8.5)
        ax.set_ylabel('Kedalaman Air Z (m)', fontsize=8.5)
        ax.set_xlim(-0.2, POOL_LEN + 0.2)
        if mode == 'SAMPING_KANAN':
            ax.invert_xaxis()
        ax.set_ylim(-POOL_DEP, 0.45)

        ax.axhline(0.0, color='#00e5ff', linestyle='-', linewidth=2.0, label='Permukaan Air')
        ax.fill_between([-0.2, POOL_LEN + 0.2], -POOL_DEP, 0.0, color='#004d66', alpha=0.25)

        # Dinding A (+X) dan Dinding C (-X)
        ax.add_patch(patches.Rectangle((4.55, -POOL_DEP), 0.25, POOL_DEP, color='#757575', label='Dinding A/C'))
        ax.add_patch(patches.Rectangle((0.20, -POOL_DEP), 0.25, POOL_DEP, color='#616161'))
        ax.plot([hx, hx + 0.12 * sim.forward_vec[0]], [-2.0, -2.0], color='#e0e0e0', linewidth=3.5)

        ref_x = [sim.wp_start[0], sim.wp_dive[0], sim.wp_approach[0], sim.wp_surface[0]]
        ref_z = [sim.wp_start[2], sim.wp_dive[2], sim.wp_approach[2], sim.wp_surface[2]]
        ax.plot(ref_x, ref_z, '--', color='#00e676', alpha=0.7, linewidth=1.6, label='Jalur Ideal')
        ax.plot(sim.traj_x, sim.traj_z, '-', color=warna_rov, linewidth=2.0, label='Jejak ROV')
        ax.plot([x, px], [z, pz], '-.', color='#00e676', alpha=0.65, linewidth=1.2, label='CV Lock')

        ax.plot([sim.wp_start[0], x], [0.25, z], ':', color='#ff9100', alpha=0.7, linewidth=1.4)
        ax.add_patch(patches.Rectangle((x - 0.20, z - 0.12), 0.40, 0.24, facecolor=warna_rov, edgecolor='white', linewidth=1.4, label='ROV MARVIC'))
        ax.add_patch(patches.Rectangle((x - 0.16, z + 0.08), 0.32, 0.06, facecolor='#ff7043'))
        ax.add_patch(patches.Circle((px, pz), 0.11, facecolor='#ff1744', edgecolor='white', linewidth=1.4, label='Payload'))
        ax.legend(loc='lower left', fontsize=7.2, ncol=3, framealpha=0.4)

    elif mode == 'ATAS':
        ax.set_title(f'TAMPAK ATAS (X - Y) | KEEMPAT SISI DINDING A, B, C, D (AKTIF: DINDING {sim.sisi})',
                     fontsize=10.0, fontweight='bold', color='#4fc3f7')
        ax.set_xlabel('Posisi X (m)', fontsize=8.5)
        ax.set_ylabel('Posisi Y (m)', fontsize=8.5)
        ax.set_xlim(-0.1, POOL_LEN + 0.1)
        ax.set_ylim(-POOL_WID / 2 - 0.15, POOL_WID / 2 + 0.15)

        ax.add_patch(patches.Rectangle((0.35, -1.85), 4.30, 3.70, color='#004d66', alpha=0.22))
        # Gambar Keempat Sisi Dinding Kolam (A = Kanan/+X, B = Atas/+Y, C = Kiri/-X, D = Bawah/-Y)
        ax.add_patch(patches.Rectangle((4.55, -1.85), 0.20, 3.70, color='#00acc1' if sim.sisi=='A' else '#616161'))
        ax.add_patch(patches.Rectangle((0.25, 1.80), 4.50, 0.18, color='#00acc1' if sim.sisi=='B' else '#616161'))
        ax.add_patch(patches.Rectangle((0.25, -1.85), 0.20, 3.70, color='#00acc1' if sim.sisi=='C' else '#616161'))
        ax.add_patch(patches.Rectangle((0.25, -1.98), 4.50, 0.18, color='#00acc1' if sim.sisi=='D' else '#616161'))

        ax.text(4.65, 0.0, 'DINDING A (+X)', rotation=90, va='center', ha='center', fontsize=7.5, fontweight='bold', color='white')
        ax.text(2.50, 1.89, 'DINDING B (+Y)', va='center', ha='center', fontsize=7.5, fontweight='bold', color='white')
        ax.text(0.35, 0.0, 'DINDING C (-X)', rotation=90, va='center', ha='center', fontsize=7.5, fontweight='bold', color='white')
        ax.text(2.50, -1.89, 'DINDING D (-Y)', va='center', ha='center', fontsize=7.5, fontweight='bold', color='white')

        ref_x = [sim.wp_start[0], sim.wp_dive[0], sim.wp_approach[0], sim.wp_surface[0]]
        ref_y = [sim.wp_start[1], sim.wp_dive[1], sim.wp_approach[1], sim.wp_surface[1]]
        ax.plot(ref_x, ref_y, '--', color='#00e676', alpha=0.7, linewidth=1.6, label='Jalur Ideal')
        ax.plot(sim.traj_x, sim.traj_y, '-', color=warna_rov, linewidth=2.0, label='Jejak ROV')
        ax.plot([x, px], [y, py], '-.', color='#00e676', alpha=0.65, linewidth=1.2, label='CV Lock')

        ax.add_patch(patches.Rectangle((x - 0.18, y - 0.18), 0.36, 0.36, facecolor=warna_rov, edgecolor='white', linewidth=1.4, label='ROV MARVIC'))
        # Panah Arah Hadap (Heading/Yaw) ROV menuju Dinding Aktif
        ax.arrow(x, y, sim.forward_vec[0] * 0.32, sim.forward_vec[1] * 0.32,
                 head_width=0.10, head_length=0.10, fc='#00e5ff', ec='#00e5ff', zorder=6)

        ax.add_patch(patches.Circle((px, py), 0.11, facecolor='#ff1744', edgecolor='white', linewidth=1.4, label='Payload'))
        ax.legend(loc='lower left', fontsize=7.2, ncol=4, framealpha=0.4)

    elif mode in ['DEPAN', 'BELAKANG']:
        judul = f'TAMPAK DEPAN (Y - Z) | TARGET: DINDING {sim.sisi}' if mode == 'DEPAN' else f'TAMPAK BELAKANG (Y - Z) | TARGET: DINDING {sim.sisi}'
        ax.set_title(judul, fontsize=10.0, fontweight='bold', color='#81c784')
        ax.set_xlabel('Posisi Lateral Y (m)', fontsize=8.5)
        ax.set_ylabel('Kedalaman Air Z (m)', fontsize=8.5)
        ax.set_xlim(-POOL_WID / 2 - 0.1, POOL_WID / 2 + 0.1)
        if mode == 'DEPAN':
            ax.invert_xaxis()
        ax.set_ylim(-POOL_DEP, 0.45)

        ax.axhline(0.0, color='#00e5ff', linestyle='-', linewidth=2.0, label='Permukaan Air')
        ax.fill_between([-POOL_WID/2 - 0.1, POOL_WID/2 + 0.1], -POOL_DEP, 0.0, color='#004d66', alpha=0.25)
        ax.add_patch(patches.Rectangle((1.80, -POOL_DEP), 0.20, POOL_DEP, color='#757575', label='Dinding B/D'))
        ax.add_patch(patches.Rectangle((-2.00, -POOL_DEP), 0.20, POOL_DEP, color='#616161'))

        ref_y = [sim.wp_start[1], sim.wp_dive[1], sim.wp_approach[1], sim.wp_surface[1]]
        ref_z = [sim.wp_start[2], sim.wp_dive[2], sim.wp_approach[2], sim.wp_surface[2]]
        ax.plot(ref_y, ref_z, '--', color='#00e676', alpha=0.7, linewidth=1.6, label='Jalur Ideal')
        ax.plot(sim.traj_y, sim.traj_z, '-', color=warna_rov, linewidth=2.0, label='Jejak ROV')

        ax.add_patch(patches.Circle((py, pz), 0.11, facecolor='#ff1744', edgecolor='white', alpha=0.85, label='Payload'))
        ax.add_patch(patches.Rectangle((y - 0.20, z - 0.12), 0.40, 0.24, facecolor=warna_rov, edgecolor='white', linewidth=1.4, label='ROV MARVIC'))
        ax.legend(loc='lower left', fontsize=7.2, ncol=3, framealpha=0.4)

def gambar_layar_3d(ax):
    ax.clear()
    ax.set_facecolor('#121212')
    ax.set_title(f'LAYAR UTAMA: 3D (30°/30°) | SKENARIO DINDING {sim.sisi} (YAW {sim.yaw_deg:+.0f}°)',
                 fontsize=10.0, fontweight='bold', color='#ce93d8')
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_zlabel('Z (m)', fontsize=8)
    ax.set_xlim(0.0, POOL_LEN)
    ax.set_ylim(-POOL_WID / 2, POOL_WID / 2)
    ax.set_zlim(-POOL_DEP, 0.4)
    ax.view_init(elev=30, azim=60)

    x, y, z = sim.pos
    px, py, pz = sim.payload_pos
    warna_rov = sim.skin_color

    wx, wy = np.meshgrid([0.0, POOL_LEN], [-POOL_WID / 2, POOL_WID / 2])
    ax.plot_surface(wx, wy, np.zeros_like(wx), color='#00e5ff', alpha=0.14)

    ref_x = [sim.wp_start[0], sim.wp_dive[0], sim.wp_approach[0], sim.wp_surface[0]]
    ref_y = [sim.wp_start[1], sim.wp_dive[1], sim.wp_approach[1], sim.wp_surface[1]]
    ref_z = [sim.wp_start[2], sim.wp_dive[2], sim.wp_approach[2], sim.wp_surface[2]]
    ax.plot(ref_x, ref_y, ref_z, '--', color='#00e676', alpha=0.75, linewidth=1.6, label='Jalur Ideal')
    ax.plot(sim.traj_x, sim.traj_y, sim.traj_z, '-', color=warna_rov, linewidth=2.0, label='Jejak ROV')
    ax.plot([x, px], [y, py], [z, pz], '-.', color='#00e676', linewidth=1.2, alpha=0.7, label='CV Lock')
    ax.plot([sim.wp_start[0], x], [sim.wp_start[1], y], [0.2, z], ':', color='#ff9100', linewidth=1.4)

    ax.scatter([x], [y], [z], color=warna_rov, edgecolor='white', s=200, marker='s', depthshade=False, label='ROV MARVIC')
    ax.scatter([px], [py], [pz], color='#ff1744', edgecolor='white', s=120, marker='o', depthshade=False, label='Payload')
    ax.legend(loc='upper left', fontsize=7.2, framealpha=0.4)

def update_frame(frame):
    if sim.screen != 'SIM':
        return

    sim.step()

    if sim.view_mode == 'ISO_3D':
        ax_main.set_visible(False)
        ax_3d.set_visible(True)
        gambar_layar_3d(ax_3d)
    elif sim.view_mode == 'POV_CAM':
        ax_3d.set_visible(False)
        ax_main.set_visible(True)
        gambar_webcam_pov(ax_main, is_main=True)
    else:
        ax_3d.set_visible(False)
        ax_main.set_visible(True)
        gambar_layar_2d(ax_main, sim.view_mode)

    gambar_webcam_pov(ax_cam, is_main=False)

    # Grafik 1 (Kiri Bawah): Posisi Horizontal & Kedalaman terhadap Waktu (Wajib sesuai PDF Tugas 4 No. 5)
    ax_pos.clear()
    ax_pos.set_title('GRAFIK POSISI & KEDALAMAN vs WAKTU', fontsize=9.0, fontweight='bold', color='#4fc3f7')
    ax_pos.set_xlabel('Waktu Tempuh (detik)', fontsize=7.8)
    ax_pos.set_ylabel('Meter (m)', fontsize=7.8)
    ax_pos.grid(True, linestyle=':', alpha=0.3)
    ax_pos.plot(sim.time_hist, sim.depth_hist, color='#00e5ff', linewidth=1.6, label=f'Kedalaman Z: {sim.pos[2]:.2f}m')
    ax_pos.plot(sim.time_hist, sim.dist_hist, color='#ffd54f', linewidth=1.6, label=f'Jarak Tempuh XY: {sim.dist_hist[-1]:.2f}m')
    ax_pos.set_xlim(0, max(5.0, sim.elapsed_time + 0.5))
    ax_pos.set_ylim(-2.6, 3.5)
    ax_pos.legend(loc='upper left', fontsize=7.3)

    # Grafik 2 (Tengah Bawah): CTE & RMSE terhadap Waktu
    ax_err.clear()
    ax_err.set_title('AKURASI JALUR: CTE & RMSE (meter)', fontsize=9.0, fontweight='bold', color='#ff8a65')
    ax_err.set_xlabel('Waktu Tempuh (detik)', fontsize=7.8)
    ax_err.set_ylabel('Error (m)', fontsize=7.8)
    ax_err.grid(True, linestyle=':', alpha=0.3)
    ax_err.plot(sim.time_hist, sim.cte_hist, color='#00e676', linewidth=1.5, label=f'CTE: {sim.cte_hist[-1]:.4f}m')
    ax_err.plot(sim.time_hist, sim.rmse_hist, color='#ff4081', linewidth=1.9, label=f'RMSE: {sim.rmse_hist[-1]:.4f}m')
    ax_err.set_xlim(0, max(5.0, sim.elapsed_time + 0.5))
    ax_err.set_ylim(0, max(0.12, max(sim.cte_hist) * 1.3))
    ax_err.legend(loc='upper right', fontsize=7.3)

    # Panel Telemetri & Log State FSM (Kanan Bawah)
    ax_hud.clear()
    ax_hud.axis('off')
    
    speed_total = float(np.linalg.norm(sim.vel))
    progress_pct = int((sim.state_idx / 6.0) * 100)
    bar_str = '█' * (progress_pct // 10) + '░' * (10 - (progress_pct // 10))
    warna_state = '#00e676' if 'GAGAL' not in sim.state else '#ff5252'
    log_str = '\n  '.join(sim.state_log[-3:])
    
    hud_text = (
        f"=== TELEMETRI & LOG STATE MISI 5 ===\n"
        f"Skenario : {sim.skenario}\n"
        f"State FSM: {sim.state} [{bar_str}] {progress_pct}%\n"
        f"Waktu    : {sim.elapsed_time:05.2f}s (Batas Misi: 600s)\n"
        f"Posisi   : X:{sim.pos[0]:+.2f} Y:{sim.pos[1]:+.2f} Z:{sim.pos[2]:+.2f}m | Yaw:{sim.yaw_deg:+.0f}°\n"
        f"Kecepatan: Vx:{sim.vel[0]:+.2f} Vy:{sim.vel[1]:+.2f} Vz:{sim.vel[2]:+.2f} (|V|:{speed_total:.2f}m/s)\n"
        f"Akurasi  : CTE:{sim.cte_hist[-1]:.4f}m | RMSE:{sim.rmse_hist[-1]:.4f}m\n"
        f"--- RIWAYAT LOG STATE (REAL-TIME) ---\n"
        f"  {log_str}"
    )
    ax_hud.text(0.0, 0.52, hud_text, transform=ax_hud.transAxes, fontsize=7.4,
                verticalalignment='center', family='monospace', color=warna_state,
                bbox=dict(boxstyle='round,pad=0.42', facecolor='#1a2327', edgecolor='#00e5ff', linewidth=1.2))


# ==========================================
# TOMBOL HALAMAN AWAL (MENU)
# ==========================================
ax_btn_mulai = plt.axes([0.38, 0.30, 0.24, 0.075])
btn_mulai = Button(ax_btn_mulai, '▶  MULAI SIMULASI', color='#00897b', hovercolor='#00bfa5')
btn_mulai.label.set_fontsize(11)
btn_mulai.label.set_fontweight('bold')
btn_mulai.label.set_color('white')

ax_btn_skin = plt.axes([0.38, 0.205, 0.24, 0.065])
btn_skin = Button(ax_btn_skin, '🎨  Ganti Skin Kapal (Kuning)', color='#1565c0', hovercolor='#1e88e5')
btn_skin.label.set_fontsize(10)
btn_skin.label.set_color('white')

ax_btn_keluar = plt.axes([0.43, 0.10, 0.14, 0.055])
btn_keluar = Button(ax_btn_keluar, '✖  KELUAR', color='#8e0000', hovercolor='#d50000')
btn_keluar.label.set_fontsize(9.5)
btn_keluar.label.set_fontweight('bold')
btn_keluar.label.set_color('white')

menu_buttons = [btn_mulai, btn_skin, btn_keluar]

# ==========================================
# TOMBOL HALAMAN SIMULASI & DROPDOWN 6 SKENARIO (4 DINDING + 2 GAGAL)
# ==========================================
sim_buttons = []
labels_views = [
    ('Samping Kiri', 'SAMPING_KIRI'),
    ('Tampak Atas', 'ATAS'),
    ('Tampak Depan', 'DEPAN'),
    ('Samping Kanan', 'SAMPING_KANAN'),
    ('Tampak Belakang', 'BELAKANG'),
    ('3D (30°/30°)', 'ISO_3D'),
    ('POV Kamera Full', 'POV_CAM'),
]

for i, (label, mode_code) in enumerate(labels_views):
    b_ax = plt.axes([0.03 + i * 0.091, 0.02, 0.086, 0.045])
    warna_btn = '#311b92' if mode_code == 'ISO_3D' else ('#006064' if mode_code == 'POV_CAM' else '#263238')
    btn = Button(b_ax, label, color=warna_btn, hovercolor='#00838f')
    btn.label.set_fontsize(7.3)
    btn.label.set_color('white')
    btn.on_clicked(lambda event, m=mode_code: setattr(sim, 'view_mode', m))
    sim_buttons.append(btn)

ax_scen = plt.axes([0.675, 0.02, 0.110, 0.045])
btn_scen = Button(ax_scen, 'Pilih Skenario ▴', color='#4a148c', hovercolor='#7b1fa2')
btn_scen.label.set_fontsize(7.5)
btn_scen.label.set_color('white')
sim_buttons.append(btn_scen)

dropdown_buttons = []
warna_opsi = ['#1b5e20', '#00695c', '#0277bd', '#283593', '#e65100', '#b71c1c']
hover_opsi = ['#2e7d32', '#00897b', '#0288d1', '#3949ab', '#ef6c00', '#c62828']

def set_dropdown_visible(visible):
    sim.dropdown_open = visible
    btn_scen.label.set_text('Tutup Pilihan ▾' if visible else 'Pilih Skenario ▴')
    for idx, b in enumerate(dropdown_buttons):
        nama_sken = DAFTAR_SKENARIO[idx]
        prefix = '✓ ' if sim.skenario == nama_sken else '   '
        b.label.set_text(f'{prefix}{nama_sken}')
        b.ax.set_visible(visible)
        b.set_active(visible)
    fig.canvas.draw_idle()

def pilih_skenario_dropdown(nama_skenario):
    sim.skenario = nama_skenario
    # Otomatis pindah ke Tampak Atas saat ganti Dinding B/D agar terlihat jelas beloknya
    if 'Dinding B' in nama_skenario or 'Dinding D' in nama_skenario:
        sim.view_mode = 'ATAS'
    sim.reset()
    set_dropdown_visible(False)

for idx, nama_sken in enumerate(DAFTAR_SKENARIO):
    d_ax = plt.axes([0.640, 0.070 + idx * 0.041, 0.175, 0.038])
    d_ax.set_zorder(25)
    d_btn = Button(d_ax, nama_sken, color=warna_opsi[idx], hovercolor=hover_opsi[idx])
    d_btn.label.set_fontsize(7.0)
    d_btn.label.set_fontweight('bold')
    d_btn.label.set_color('white')
    d_btn.on_clicked(lambda event, s=nama_sken: pilih_skenario_dropdown(s))
    d_ax.set_visible(False)
    d_btn.set_active(False)
    dropdown_buttons.append(d_btn)

btn_scen.on_clicked(lambda event: set_dropdown_visible(not sim.dropdown_open))

ax_rst = plt.axes([0.793, 0.02, 0.088, 0.045])
btn_rst = Button(ax_rst, 'Reset / Ulang', color='#b71c1c', hovercolor='#e53935')
btn_rst.label.set_fontsize(7.5)
btn_rst.label.set_color('white')
btn_rst.on_clicked(lambda event: sim.reset())
sim_buttons.append(btn_rst)

ax_back = plt.axes([0.889, 0.02, 0.080, 0.045])
btn_back = Button(ax_back, 'Menu Awal', color='#37474f', hovercolor='#546e7a')
btn_back.label.set_fontsize(7.5)
btn_back.label.set_color('white')
sim_buttons.append(btn_back)

# ==========================================
# NAVIGASI ANTAR HALAMAN (MENU <-> SIMULASI)
# ==========================================
def set_screen(target_screen):
    sim.screen = target_screen
    is_menu = (target_screen == 'MENU')

    if is_menu:
        set_dropdown_visible(False)

    ax_menu.set_visible(is_menu)
    for b in menu_buttons:
        b.ax.set_visible(is_menu)
        b.set_active(is_menu)

    sim_title.set_visible(not is_menu)
    ax_cam.set_visible(not is_menu)
    ax_pos.set_visible(not is_menu)
    ax_err.set_visible(not is_menu)
    ax_hud.set_visible(not is_menu)
    if is_menu:
        ax_main.set_visible(False)
        ax_3d.set_visible(False)
        gambar_menu()
    else:
        sim.reset()
        ax_main.set_visible(sim.view_mode != 'ISO_3D')
        ax_3d.set_visible(sim.view_mode == 'ISO_3D')

    for b in sim_buttons:
        b.ax.set_visible(not is_menu)
        b.set_active(not is_menu)

    fig.canvas.draw_idle()

def on_ganti_skin(event):
    sim.next_skin()
    btn_skin.label.set_text(f'🎨  Ganti Skin Kapal ({sim.skin_name})')
    gambar_menu()

btn_mulai.on_clicked(lambda event: set_screen('SIM'))
btn_skin.on_clicked(on_ganti_skin)
btn_keluar.on_clicked(lambda event: plt.close(fig))
btn_back.on_clicked(lambda event: set_screen('MENU'))

set_screen('MENU')

ani = FuncAnimation(fig, update_frame, interval=int(DT * 1000), cache_frame_data=False)
plt.show()