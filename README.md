# ⚽ Football Matches Analysis with YOLOv11

This repository contains a deep learning-based system for analyzing **indoor football matches** using **YOLOv11** for object detection and tracking. The model is **fine-tuned on a custom indoor football dataset** that I personally annotated, enabling it to **accurately detect players, the ball, and referees in an indoor football setting**.

The project **tracks players, estimates ball possession, assigns team colors, calculates player speed & distance, and generates an annotated video** with all this information visualized.

🎥 **Example Output:**
![Watch the video](https://raw.githubusercontent.com/FdiLmr/ML-Football-Analysis/main/examples/output.gif)

---

## 🚧 Work in Progress
This project is **actively being improved**! Here’s what’s coming next:

- **⚽ Enhanced Ball Detection:** The current model will be refined for better ball tracking accuracy.
- **🛠 More Functionalities:** Future updates will include automatic event detection (goals, assists, tackles, etc.).
- **🎨 UI Improvements:** A more polished visualization with better overlays and real-time analytics.
- **☁️ Cloud Processing:** A potential web-based platform to analyze matches without local setup.

Contributions & feedback are welcome! 🚀

---

## 🚀 Project Overview

This project implements a **video processing pipeline** that **detects, tracks, and analyzes** objects in an indoor football match. Below are the main steps:

1. **Video Input** 🎥  
   The system loads a football match video frame-by-frame.

2. **Object Detection & Tracking** 🔍  
   Uses a **fine-tuned YOLOv11 model** to detect and track **players, referees, and the ball**.

3. **Player Position Estimation** 📍  
   Extracts **precise locations** of players and the ball to understand movement dynamics.

4. **Ball Position Interpolation** ⚽  
   Fills in missing ball positions to ensure smooth tracking.

5. **Speed & Distance Calculation** ⏱️  
   Estimates **player and ball speed** (in km/h) and **distance covered** (in meters) throughout the game.

6. **Team Assignment** 🔴🔵  
   Automatically identifies **team colors** based on player positions and interactions.

7. **Ball Possession Detection** ⚡  
   Determines **which player has control of the ball** at any given moment.

8. **Visualization & Output Video** 🎨  
   The system overlays annotations onto each frame, displaying:
   - Player IDs  
   - Team colors  
   - Ball possession  
   - Speed & distance traveled  
   
   Finally, all annotated frames are **compiled into a video output**.

---

## 🏗 Technologies Used
- **YOLOv11** 🚀 - Object detection, fine-tuned on a **custom indoor football dataset**  
- **Roboflow** 🏷️ - Used for dataset annotation and management  
- **Ultralytics** 📡 - Implements YOLOv11 for training & evaluation  
- **OpenCV** 👀 - Video processing & frame manipulation  
- **NumPy** 🔢 - Numerical computations  
- **Matplotlib** 📊 - Data visualization  
- **Pandas** 📑 - Data handling and analytics  
- **FFmpeg** 🎬 - Video input/output processing  
- **PyTorch** 🔥 - Model training & inference  

---

## 📈 Future Enhancements
- 🏃 **Player Performance Metrics** → Heatmaps, shot accuracy, passing efficiency  
- 🖥️ **User Interface** → A web-based tool to upload & analyze videos without coding  
- 🎯 **Custom Model Training** → Allow users to fine-tune on their own datasets  
- ☁️ **Cloud Deployment** → Online processing for scalable video analysis  

---

## 🛠 How to Run the Project
### 1️⃣ Clone this repository:
```bash
git clone https://github.com/yourusername/football-matches-analysis.git
cd football-matches-analysis

### 2️⃣  Install dependencies:
pip install -r requirements.txt

### 3️⃣ Run the analysis:
python main.py