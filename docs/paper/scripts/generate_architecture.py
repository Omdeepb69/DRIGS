"""
DRIGS Professional Architecture Diagram Generator
Produces publication-quality, crisp, modern IEEE-style system architecture figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib

# Set up matplotlib font rendering
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Courier New']
plt.rcParams['text.usetex'] = False

# ── Color Palette ─────────────────────────────────────────────────────────────
C_BG            = "#FFFFFF"
C_TEXT_MAIN     = "#0F172A"   # Slate 900
C_TEXT_MUTED    = "#475569"   # Slate 600
C_TEXT_LIGHT    = "#64748B"   # Slate 500

# Layer 1: Control Plane (Navy / Indigo)
CP_BAND_BG      = "#F8FAFC"   # Slate 50
CP_BORDER       = "#CBD5E1"   # Slate 300
CP_HEADER_BG    = "#1E293B"   # Slate 800
CP_CARD_BG      = "#FFFFFF"
CP_CARD_BORDER  = "#4F46E5"   # Indigo 600
CP_TEXT_TITLE   = "#1E1B4B"   # Indigo 950

# Layer 2: Worker Nodes (Sky / Cyan)
WN_BAND_BG      = "#F0F9FF"   # Sky 50
WN_BORDER       = "#BAE6FD"   # Sky 200
WN_HEADER_BG    = "#0284C7"   # Sky 600
WN_CARD_BG      = "#FFFFFF"
WN_CARD_BORDER  = "#0369A1"   # Sky 700

# Layer 3: Execution Layer (Emerald / Green)
EX_BAND_BG      = "#F0FDF4"   # Emerald 50
EX_BORDER       = "#BBF7D0"   # Emerald 200
EX_HEADER_BG    = "#15803D"   # Emerald 700
EX_CARD_BG      = "#FFFFFF"
EX_CARD_BORDER  = "#16A34A"   # Emerald 600

# Accents
REST_CARD_BG    = "#FFFBEB"   # Amber 50
REST_BORDER     = "#D97706"   # Amber 600
REST_TEXT       = "#92400E"   # Amber 800

SQL_CARD_BG     = "#FFF1F2"   # Rose 50
SQL_BORDER      = "#E11D48"   # Rose 600
SQL_TEXT        = "#9F1239"   # Rose 800

C_ARROW_MAIN    = "#334155"   # Slate 700
C_ARROW_CTRL    = "#4F46E5"   # Indigo 600
C_ARROW_WORK    = "#0284C7"   # Sky 600
C_ARROW_EXEC    = "#16A34A"   # Emerald 600

# ── Canvas Setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=300)
ax.set_xlim(0, 10.8)
ax.set_ylim(0, 6.0)
ax.axis("off")
fig.patch.set_facecolor(C_BG)

# ── Helper Drawing Functions ──────────────────────────────────────────────────

def draw_card(x, y, w, h, title, subtitle=None, badge=None, bg=CP_CARD_BG, border=CP_CARD_BORDER, title_color=C_TEXT_MAIN, corner_r=0.07):
    """Draws a component card with guaranteed non-overlapping text layout."""
    card = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={corner_r}",
        linewidth=1.1, edgecolor=border, facecolor=bg, zorder=3
    )
    ax.add_patch(card)
    
    if subtitle and badge:
        y_title = y + h - 0.22
        y_sub   = y + h - 0.44
        y_badge = y + h - 0.65
    elif subtitle:
        y_title = y + h - 0.26
        y_sub   = y + h - 0.52
        y_badge = None
    else:
        y_title = y + h / 2
        y_sub   = None
        y_badge = None

    ax.text(x + w/2, y_title, title, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=title_color, zorder=4)
    
    if subtitle:
        ax.text(x + w/2, y_sub, subtitle, ha="center", va="center",
                fontsize=5.8, color=C_TEXT_MUTED, zorder=4)
        
    if badge:
        ax.text(x + w/2, y_badge, badge, ha="center", va="center",
                fontsize=5.1, fontfamily="monospace", color=C_TEXT_LIGHT, zorder=4)

def draw_band_header(x, y, w, h, label, bg_color):
    """Draws a vertical header tab on the left of each band."""
    header = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=0, facecolor=bg_color, zorder=3
    )
    ax.add_patch(header)
    ax.text(x + w/2, y + h/2, label.upper(), ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#FFFFFF", zorder=4,
            rotation=90)

def draw_arrow(x0, y0, x1, y1, label=None, linestyle="-", color=C_ARROW_MAIN, width=1.0, rad=0.0, label_pos=0.5):
    """Draws a directional arrow with clean label placement."""
    connectionstyle = f"arc3,rad={rad}"
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            linestyle=linestyle,
            color=color,
            lw=width,
            mutation_scale=8,
            connectionstyle=connectionstyle
        ),
        zorder=5
    )
    if label:
        lx = x0 + (x1 - x0) * label_pos
        ly = y0 + (y1 - y0) * label_pos
        ax.text(
            lx, ly, label, ha="center", va="center",
            fontsize=5.3, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor=color, lw=0.6, alpha=0.95),
            zorder=6
        )

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(5.4, 5.82, "DRIGS Three-Layer System Architecture", ha="center", va="center",
        fontsize=11.0, fontweight="bold", color=C_TEXT_MAIN, zorder=6)
ax.text(5.4, 5.62, "Decoupled Control Plane · Dynamic Worker Discovery · Modular Execution Backends",
        ha="center", va="center", fontsize=6.8, color=C_TEXT_MUTED, zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: CONTROL PLANE (y: 3.70 -> 5.45)
# ══════════════════════════════════════════════════════════════════════════════
CP_Y, CP_H = 3.70, 1.75
cp_bg = mpatches.FancyBboxPatch(
    (0.1, CP_Y), 10.6, CP_H,
    boxstyle="round,pad=0,rounding_size=0.08",
    linewidth=1.0, edgecolor=CP_BORDER, facecolor=CP_BAND_BG, zorder=1
)
ax.add_patch(cp_bg)
draw_band_header(0.1, CP_Y, 0.45, CP_H, "Control Plane", CP_HEADER_BG)

y_row1 = 4.45
h_card = 0.84

# 1. REST API
draw_card(0.68, y_row1, 1.25, h_card, "REST API Gateway", "CLI / HTTP Client", "POST /v1/jobs",
          bg=REST_CARD_BG, border=REST_BORDER, title_color=REST_TEXT)

# 2. Admission
draw_card(2.05, y_row1, 1.25, h_card, "Admission Ctrl", "Validation & Auth", "Quota Check",
          bg=CP_CARD_BG, border=CP_CARD_BORDER, title_color=CP_TEXT_TITLE)

# 3. Job Queue
draw_card(3.42, y_row1, 1.25, h_card, "Job Queue", "Priority & Fair Share", "FIFO / DRF",
          bg=CP_CARD_BG, border=CP_CARD_BORDER, title_color=CP_TEXT_TITLE)

# 4. Scheduler Engine
draw_card(4.79, y_row1, 1.45, h_card, "Scheduler Engine", "Pluggable Policy Core", "FIFO · DRF · Gang · Topo",
          bg=CP_CARD_BG, border=CP_CARD_BORDER, title_color=CP_TEXT_TITLE)

# 5. Worker Registry
draw_card(6.36, y_row1, 1.35, h_card, "Worker Registry", "Lifecycle & Heartbeats", "ONLINE / DEGRADED",
          bg=CP_CARD_BG, border=CP_CARD_BORDER, title_color=CP_TEXT_TITLE)

# 6. Failure Detector
draw_card(7.83, y_row1, 1.35, h_card, "Failure Detector", "Rescheduler Controller", "Zombie Sweeper & Retry",
          bg=CP_CARD_BG, border=CP_CARD_BORDER, title_color=CP_TEXT_TITLE)

# 7. SQLite Store
draw_card(9.30, y_row1, 1.30, h_card, "SQLite Store", "ACID State Engine", "Jobs · Workers · State",
          bg=SQL_CARD_BG, border=SQL_BORDER, title_color=SQL_TEXT)

# Horizontal Pipeline Arrows
draw_arrow(1.93, y_row1 + 0.42, 2.05, y_row1 + 0.42, color=CP_CARD_BORDER)
draw_arrow(3.30, y_row1 + 0.42, 3.42, y_row1 + 0.42, color=CP_CARD_BORDER)
draw_arrow(4.67, y_row1 + 0.42, 4.79, y_row1 + 0.42, color=CP_CARD_BORDER)
draw_arrow(6.24, y_row1 + 0.42, 6.36, y_row1 + 0.42, color=CP_CARD_BORDER)
draw_arrow(7.71, y_row1 + 0.42, 7.83, y_row1 + 0.42, color=CP_CARD_BORDER)
draw_arrow(9.18, y_row1 + 0.42, 9.30, y_row1 + 0.42, color=SQL_BORDER)

# Control Plane Protocol Banner
cp_banner = mpatches.FancyBboxPatch(
    (0.68, CP_Y + 0.12), 9.92, 0.45,
    boxstyle="round,pad=0,rounding_size=0.04",
    linewidth=0.8, edgecolor="#CBD5E1", facecolor="#EEF2FF", zorder=2
)
ax.add_patch(cp_banner)
ax.text(5.64, CP_Y + 0.35, "Abstract Protocol Contracts:  HardwareBackend  ·  Scheduler  ·  ExecutionBackend  ·  WorkerRegistryProtocol",
        ha="center", va="center", fontsize=6.2, fontfamily="monospace", fontweight="bold", color=CP_CARD_BORDER, zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: WORKER NODES (y: 1.80 -> 3.25)
# ══════════════════════════════════════════════════════════════════════════════
WN_Y, WN_H = 1.80, 1.45
wn_bg = mpatches.FancyBboxPatch(
    (0.1, WN_Y), 10.6, WN_H,
    boxstyle="round,pad=0,rounding_size=0.08",
    linewidth=1.0, edgecolor=WN_BORDER, facecolor=WN_BAND_BG, zorder=1
)
ax.add_patch(wn_bg)
draw_band_header(0.1, WN_Y, 0.45, WN_H, "Worker Nodes", WN_HEADER_BG)

# Worker Nodes
w_xs = [0.68, 4.00, 7.32]
w_w = 2.80
w_h = 1.12

workers = [
    ("Worker Agent Node 0", "NVML Telemetry & Process Guard", "NVML Telemetry • VRAM Tracker • Heartbeat"),
    ("Worker Agent Node 1", "NVML Telemetry & Process Guard", "NVML Telemetry • VRAM Tracker • Heartbeat"),
    (r"Worker Agent Node $M-1$", "NVML Telemetry & Process Guard", "NVML Telemetry • VRAM Tracker • Heartbeat")
]

for wx, (w_title, w_sub, w_badge) in zip(w_xs, workers):
    draw_card(wx, WN_Y + 0.16, w_w, w_h, w_title, w_sub, w_badge,
              bg=WN_CARD_BG, border=WN_CARD_BORDER, title_color=WN_HEADER_BG)

# Ellipsis between Worker 1 and Worker M-1
ax.text(6.96, WN_Y + WN_H/2, "•  •  •", ha="center", va="center",
        fontsize=14, color=WN_HEADER_BG, fontweight="bold", zorder=4)

# ── Inter-Layer Control <-> Worker Connections ────────────────────────────────
# 1. Dispatch Job Allocation (Scheduler -> Worker 0)
draw_arrow(2.68, CP_Y, 2.08, WN_Y + WN_H, label="Dispatch Job Allocation", color=CP_CARD_BORDER, width=1.1)

# 2. State Command (Scheduler -> Worker 1)
draw_arrow(5.50, CP_Y, 5.40, WN_Y + WN_H, label="Scheduler State Command", color=CP_CARD_BORDER, width=1.1)

# 3. Heartbeat & Telemetry (Worker M-1 -> Worker Registry)
draw_arrow(8.72, WN_Y + WN_H, 7.03, CP_Y, label="POST /v1/workers/heartbeat (15s Sweep)", linestyle="--", color=WN_HEADER_BG, width=1.1)

# ══════════════════════════════════════════════════════════════════════════════
# INTER-LAYER WORKER <-> EXECUTION CONNECTIONS
# ══════════════════════════════════════════════════════════════════════════════
draw_arrow(2.08, WN_Y, 2.15, 1.45, label="Subprocess Driver", color=EX_HEADER_BG, width=1.0)
draw_arrow(5.40, WN_Y, 5.47, 1.45, label="Container Runtime", color=EX_HEADER_BG, width=1.0)
draw_arrow(8.72, WN_Y, 8.80, 1.45, label="Gang Rank Sync", color=EX_HEADER_BG, width=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: EXECUTION BACKENDS (y: 0.10 -> 1.45)
# ══════════════════════════════════════════════════════════════════════════════
EX_Y, EX_H = 0.10, 1.35
ex_bg = mpatches.FancyBboxPatch(
    (0.1, EX_Y), 10.6, EX_H,
    boxstyle="round,pad=0,rounding_size=0.08",
    linewidth=1.0, edgecolor=EX_BORDER, facecolor=EX_BAND_BG, zorder=1
)
ax.add_patch(ex_bg)
draw_band_header(0.1, EX_Y, 0.45, EX_H, "Execution Layer", EX_HEADER_BG)

# Execution Backends
e_xs = [0.68, 4.00, 7.32]
e_w = 2.95
e_h = 1.05

backends = [
    ("NativeProcessBackend", "Subprocess Driver & Guard", "CUDA_VISIBLE_DEVICES • psutil Tree Teardown"),
    ("DockerBackend", "Isolated Container Runtime", "nvidia-docker • Resource Limits • Volume Mounts"),
    ("DistributedBackend", "Multi-Node Gang Sync", "PyTorch DDP • torchrun • GangScheduler")
]

for ex, (b_title, b_sub, b_badge) in zip(e_xs, backends):
    draw_card(ex, EX_Y + 0.15, e_w, e_h, b_title, b_sub, b_badge,
              bg=EX_CARD_BG, border=EX_CARD_BORDER, title_color=EX_HEADER_BG)

# Save output
plt.tight_layout(pad=0.1)
out_dir = pathlib.Path(__file__).parent.parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "architecture.pdf"

fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Generated PDF: {pdf_path}")
