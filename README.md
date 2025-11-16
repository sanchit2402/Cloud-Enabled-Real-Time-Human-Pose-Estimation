

# **Cloud-Enabled Real-Time Human Pose Estimation**

A scalable, real-time human pose estimation system powered by cloud inference, computer vision, and deep learning. This project processes video streams (webcam / uploaded videos) in real-time and returns accurate 2D/3D human pose keypoints using a cloud-hosted AI model.

---

## 🚀 **Features**

* **Real-time pose estimation** using cloud inference (Azure / AWS / GCP).
* **Supports images, videos, and live webcam streams.**
* **High-accuracy detection** using Mediapipe / OpenPose / MoveNet (your selected model).
* **Low-latency streaming pipeline** optimized for edge-to-cloud transmission.
* **Scalable microservice architecture** (API gateway + inference server).
* **Built-in FPS counter and performance monitor.**
* **Web-based UI (Streamlit)** for easy interaction.
* **Optional GPU acceleration** for high performance.

---

## 🧠 **System Architecture**

```
User Device (Webcam / Video File)
        │
        ▼
Frontend UI (Streamlit / Web App)
        │  Compressed frames
        ▼
Cloud API Gateway
        │
        ▼
Inference Engine (Cloud GPU / Container)
        │  Keypoints JSON
        ▼
Frontend (Skeleton Rendering)
```

---

## ⚙️ **Tech Stack**

### **Frontend**

* Python
* Streamlit
* OpenCV
* WebRTC / VideoCapture

### **Backend / Cloud**

* Azure ML / Azure Functions / Azure Container Apps *(or AWS/GCP equivalent)*
* FastAPI (API server)
* Docker container for inference
* GPU-enabled environment (NVIDIA CUDA)

### **Pose Models Supported**

* MediaPipe Pose
* MoveNet (Thunder/Lightning)
* OpenPose (Optional)
* BlazePose

---

## 🛠️ **Installation**

### **Clone the Repository**

```bash
git clone https://github.com/sanchit2402/Cloud-Enabled-Real-Time-Human-Pose-Estimation.git
cd Cloud-Enabled-Real-Time-Human-Pose-Estimation
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Run Locally**

```bash
 python -m streamlit run app_streamlit.py
```

---


## ▶️ **Usage**

### **Live Webcam Pose Tracking**

* Open the Streamlit UI
* Click **"Start Webcam"**
* Pose landmarks and skeleton will appear live

### **Video File**

* Upload any `.mp4` file
* Server processes each frame using cloud inference

### **Image Mode**

* Upload an image
* Results returned instantly from cloud

---

## 🎯 **Key Applications**

* Fitness and workout tracking
* Yoga posture correction
* Sports analytics
* Physiotherapy
* Surveillance & activity recognition
* Human-computer interaction (HCI)
* Motion analysis

---

## 📊 **Performance**

| Model             | Latency  | Accuracy  | Best Use      |
| ----------------- | -------- | --------- | ------------- |
| MoveNet Lightning | ~10ms    | Medium    | High FPS apps |
| MoveNet Thunder   | ~15–20ms | High      | Fitness apps  |
| MediaPipe Pose    | ~20–30ms | High      | Balanced      |
| OpenPose          | 40–60ms  | Very High | Research      |

---


## 🤝 **Contributing**

Pull requests are welcome!
Please follow the standard PR template and coding guidelines.

---

## 📜 **License**

MIT License. Free to use and modify.

---

## ⭐ **Support the Project**

If you find this project useful, consider giving it a ⭐ on GitHub.

---


