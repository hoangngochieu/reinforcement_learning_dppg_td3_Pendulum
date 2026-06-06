# 🎯 DDPG vs TD3 — Overestimation Bias Comparison on Pendulum-v1

> **🌐 Language / Ngôn ngữ:** [English](#-english) | [Tiếng Việt](#-tiếng-việt)

---

# 🇬🇧 English

This project implements and compares two popular **Reinforcement Learning** algorithms:

- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed DDPG)

on the **Pendulum-v1** environment from Gymnasium, focusing on analyzing **Overestimation Bias** — the problem of overestimating Q-values in DDPG and how TD3 addresses it.

---


## ⚙️ System Requirements

- **Python**: 3.8 or higher (3.10+ recommended)
- **OS**: Windows / Linux / macOS
- **GPU** *(optional)*: CUDA-compatible GPU will speed up training

---

## 📦 Installation Guide

### Step 1: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (CMD):
.venv\Scripts\activate.bat

# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Required Libraries

```bash
pip install numpy torch gymnasium matplotlib
```

#### Library Details:

| Library        | Purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| `numpy`        | Array operations, matrix computations                         |
| `torch`        | PyTorch — deep learning framework for Actor/Critic networks   |
| `gymnasium`    | Pendulum-v1 simulation environment (successor to OpenAI Gym)  |
| `matplotlib`   | Plotting comparison charts                                    |

> **💡 Note on PyTorch with GPU:**
> If you have an NVIDIA GPU and want to accelerate training, install the CUDA version of PyTorch instead of the default CPU version.
> Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for the appropriate installation command.
>
> Example for CUDA 12.1:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 3: Verify Installation

```bash
python -c "import torch; import gymnasium; import numpy; import matplotlib; print('✅ All libraries installed successfully!')"
```

---

## 🚀 How to Run

The project has **2 main Python files**, each serving a different purpose:

---

### 📊 File 1: `compare_overestimation.py` — Train & Plot Charts

**Purpose:** Train both DDPG and TD3 from scratch, then plot 4 comparison charts.

```bash
python compare_overestimation.py
```

**What happens:**
1. Trains DDPG on Pendulum-v1 (100,000 timesteps)
2. Trains TD3 on Pendulum-v1 (100,000 timesteps)
3. Automatically plots and saves chart as `td3_vs_ddpg_overestimation.png`

**Output:**
- 4 charts: Overestimation Bias, Estimated Q vs True Q, Episode Reward, Average Bias
- Estimated runtime: **5–15 minutes** (depending on CPU/GPU)

---

### 🎮 File 2: `compare_vizualize.py` — Train + Visual Demo

**Purpose:** Train models, save weights, and watch agents perform visually (render).

This file has **2 modes** controlled by 2 flags in the code:

```python
TRAIN_MODE = True       # True = Train and save models
VISUALIZE_MODE = True   # True = Watch agents perform visually
```

#### Mode 1: First-time Training

Open `compare_vizualize.py` and set:
```python
TRAIN_MODE = True
VISUALIZE_MODE = False
```

Then run:
```bash
python compare_vizualize.py
```

Result: Saves 4 `.pth` weight files + comparison charts.

#### Mode 2: Visualization Only (after training)

Set:
```python
TRAIN_MODE = False
VISUALIZE_MODE = True
```

Then run:
```bash
python compare_vizualize.py
```

Result: Opens a Pygame window showing the Pendulum controlled by DDPG and TD3.

#### Mode 3: Train + Visualize

Set:
```python
TRAIN_MODE = True
VISUALIZE_MODE = True
```

```bash
python compare_vizualize.py
```

> **⚠️ Note:** Pre-trained `.pth` weight files are already included in the repo. You can run Visualization mode immediately without training.

---

## 📈 Understanding the Charts

After training, the program displays 4 charts:

| Chart | Description |
|-------|-------------|
| **Overestimation Bias** | Compares Q-value overestimation levels (DDPG is typically higher than TD3) |
| **Estimated Q vs True Q** | Solid line = estimated Q, dashed line = true Q (Monte Carlo rollout) |
| **Episode Reward** | Learning performance across episodes (TD3 is typically more stable) |
| **Average Bias** | Bar chart comparing the mean bias of both algorithms |

---

## 🧠 Algorithm Summary

### DDPG (Deep Deterministic Policy Gradient)
- Uses **1 Critic** only → prone to **overestimating** Q-values
- Updates Actor **every step**

### TD3 (Twin Delayed DDPG) — 3 Key Improvements:
1. **Clipped Double-Q**: Uses **2 Critics**, takes `min(Q1, Q2)` → reduces overestimation
2. **Delayed Policy Updates**: Updates Actor **every 2 steps** instead of every step
3. **Target Policy Smoothing**: Adds noise to target actions → prevents exploiting local errors

---


## 👨‍💻 Author

- **Hoàng Ngọc Hiếu**
- Email: hoangngochieutin92018@gmail.com

---
---

# 🇻🇳 Tiếng Việt

Dự án này triển khai và so sánh hai thuật toán **Reinforcement Learning** nổi tiếng:

- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed DDPG)

trên môi trường **Pendulum-v1** của Gymnasium, tập trung phân tích **Overestimation Bias** — vấn đề ước lượng Q-value quá cao trong DDPG và cách TD3 khắc phục.

---

## 📁 Cấu trúc dự án

```
reinforcement_learning/
├── compare_overestimation.py   # Train DDPG & TD3, vẽ biểu đồ so sánh Overestimation Bias
├── compare_vizualize.py        # Train + Visualize (xem agent hoạt động trực quan)
├── giai thich code.docx        # Tài liệu giải thích chi tiết code
├── DDPG_model_actor.pth        # Trọng số Actor đã train (DDPG)
├── DDPG_model_critic.pth       # Trọng số Critic đã train (DDPG)
├── TD3_model_actor.pth         # Trọng số Actor đã train (TD3)
├── TD3_model_critic.pth        # Trọng số Critic đã train (TD3)
├── td3_vs_ddpg_overestimation.png  # Biểu đồ kết quả so sánh
├── .gitignore
└── README.md                   # File này
```

---

## ⚙️ Yêu cầu hệ thống

- **Python**: 3.8 trở lên (khuyến nghị 3.10+)
- **Hệ điều hành**: Windows / Linux / macOS
- **GPU** *(tùy chọn)*: Có CUDA-compatible GPU sẽ tăng tốc huấn luyện

---

## 📦 Hướng dẫn cài đặt thư viện

### Bước 1: Tạo môi trường ảo (Virtual Environment)

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt môi trường ảo
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (CMD):
.venv\Scripts\activate.bat

# Linux / macOS:
source .venv/bin/activate
```

### Bước 2: Cài đặt các thư viện cần thiết

```bash
pip install numpy torch gymnasium matplotlib
```

#### Chi tiết các thư viện:

| Thư viện       | Mục đích                                                     |
| -------------- | ------------------------------------------------------------ |
| `numpy`        | Xử lý mảng số, tính toán ma trận                            |
| `torch`        | PyTorch — framework deep learning, xây dựng mạng Actor/Critic |
| `gymnasium`    | Môi trường mô phỏng Pendulum-v1 (thay thế OpenAI Gym cũ)    |
| `matplotlib`   | Vẽ biểu đồ so sánh kết quả                                  |

> **💡 Lưu ý về PyTorch với GPU:**
> Nếu bạn có GPU NVIDIA và muốn tăng tốc huấn luyện, hãy cài PyTorch phiên bản CUDA thay vì bản CPU mặc định.
> Truy cập [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) để lấy lệnh cài đặt phù hợp.
>
> Ví dụ cho CUDA 12.1:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Bước 3: Kiểm tra cài đặt

```bash
python -c "import torch; import gymnasium; import numpy; import matplotlib; print('✅ Tất cả thư viện đã được cài đặt thành công!')"
```

---

## 🚀 Hướng dẫn chạy code

Dự án có **2 file Python** chính, mỗi file phục vụ mục đích khác nhau:

---

### 📊 File 1: `compare_overestimation.py` — Train và vẽ biểu đồ

**Mục đích:** Huấn luyện cả DDPG và TD3 từ đầu, sau đó vẽ 4 biểu đồ so sánh.

```bash
python compare_overestimation.py
```

**Quá trình chạy:**
1. Huấn luyện DDPG trên Pendulum-v1 (100,000 timesteps)
2. Huấn luyện TD3 trên Pendulum-v1 (100,000 timesteps)
3. Tự động vẽ và lưu biểu đồ `td3_vs_ddpg_overestimation.png`

**Kết quả đầu ra:**
- 4 biểu đồ: Overestimation Bias, Estimated Q vs True Q, Episode Reward, Trung bình Bias
- Thời gian chạy ước tính: **5–15 phút** (tùy CPU/GPU)

---

### 🎮 File 2: `compare_vizualize.py` — Train + Xem trực quan

**Mục đích:** Huấn luyện model, lưu trọng số, và xem agent hoạt động trực quan (render).

File này có **2 chế độ** được điều khiển bởi 2 biến cờ trong code:

```python
TRAIN_MODE = True       # True = Huấn luyện và lưu model
VISUALIZE_MODE = True   # True = Xem agent hoạt động trực quan
```

#### Chế độ 1: Huấn luyện lần đầu

Mở file `compare_vizualize.py`, đặt:
```python
TRAIN_MODE = True
VISUALIZE_MODE = False
```

Rồi chạy:
```bash
python compare_vizualize.py
```

Kết quả: Lưu 4 file trọng số `.pth` + biểu đồ so sánh.

#### Chế độ 2: Xem Visualize (sau khi đã train)

Đặt:
```python
TRAIN_MODE = False
VISUALIZE_MODE = True
```

Rồi chạy:
```bash
python compare_vizualize.py
```

Kết quả: Mở cửa sổ Pygame hiển thị con lắc Pendulum được điều khiển bởi DDPG và TD3.

#### Chế độ 3: Vừa Train vừa Visualize

Đặt:
```python
TRAIN_MODE = True
VISUALIZE_MODE = True
```

```bash
python compare_vizualize.py
```

> **⚠️ Lưu ý:** Các file `.pth` (trọng số model) đã được train sẵn và có trong repo. Bạn có thể chạy ngay chế độ Visualize mà không cần train lại.

---

## 📈 Giải thích biểu đồ kết quả

Sau khi train xong, chương trình sẽ hiển thị 4 biểu đồ:

| Biểu đồ | Ý nghĩa |
|----------|----------|
| **Overestimation Bias** | So sánh mức độ ước lượng Q quá cao (DDPG thường cao hơn TD3) |
| **Estimated Q vs True Q** | Đường liền = Q ước lượng, đường đứt = Q thực tế (Monte Carlo) |
| **Episode Reward** | Hiệu suất học tập qua từng episode (TD3 thường ổn định hơn) |
| **Trung bình Bias** | Biểu đồ cột so sánh bias trung bình của 2 thuật toán |

---

## 🧠 Tóm tắt thuật toán

### DDPG (Deep Deterministic Policy Gradient)
- Dùng **1 Critic** duy nhất → dễ bị **overestimate** Q-value
- Cập nhật Actor **mỗi step**

### TD3 (Twin Delayed DDPG) — 3 cải tiến chính:
1. **Clipped Double-Q**: Dùng **2 Critics**, lấy `min(Q1, Q2)` → giảm overestimation
2. **Delayed Policy Updates**: Cập nhật Actor **mỗi 2 steps** thay vì mỗi step
3. **Target Policy Smoothing**: Thêm noise vào target action → tránh exploit lỗi cục bộ

---

## 👨‍💻 Tác giả

- **Hoàng Ngọc Hiếu**
- Email: hoangngochieutin92018@gmail.com
