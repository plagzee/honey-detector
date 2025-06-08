# 🐶 Honey Detector

A simple Python app that uses [OpenCV](https://opencv.org) and PyTorch to detect **Honey** – a specific pet dog – in real-time video feed using your webcam! 🎥🐾

This project demonstrates the use of **Machine Learning** and **Computer Vision** to distinguish Honey from other dogs, humans, or objects in live camera input.

---

## 📚 Contents
- [🔍 Overview](#-overview)
- [⚙️ Installation](#-installation)
- [📱 Contribute (Android App)](#-contribute-android-app)
- [🚧 To Do](#-to-do)

---

## 🔍 Overview

### ❌ Honey Not Detected
![image](nothoney.png)

If Honey is **not** in the frame, the app shows a "Not Honey!" message.

---

### ✅ Honey Detected
![image](honey.png)

If Honey **is** detected, the app proudly announces: **"Honey!"**

This app is currently a Python script that runs on your local machine using `OpenCV`. Future plans include building a **web interface using Flask** (target date: 08-06-2025).

---

## 🧠 What's Inside

### 📦 `requirements.txt`
Lists all Python packages needed to run the project.

### 🖼️ `downloaded_scripts/download_images.py`
Used to download **non-Honey** images from the internet using `simple_image_download`.  
(Note: The `simple_images/` folder is ignored by Git.)

### 🏋️‍♂️ `train.py`
Trains a **binary classifier** model to detect Honey vs. Not Honey.  
Outputs the trained model as `honey_detector_model.pt` and also plots training loss and accuracy.


![Train mode](matplot.png)


### 🔍 `predict.py`
A script to run predictions on test images using the trained model.

### 🎥 `webcam.py` & `webcam_detect.py`
These scripts run real-time honey detection using your webcam.  
They demonstrate how the trained model is integrated into a live video feed using OpenCV.

> 💡 **Note**: Future plans include creating a web version using Flask, or a desktop app using Tkinter.

---

## ⚙️ Installation

As of **08-06-2025**, the app is executable only via **Python** + **OpenCV**.

### 🔧 Steps to Get Started
1. 🧬 Clone the repository:
   ```bash
   git clone https://github.com/your-username/honey-detector.git

2. 📦 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. 🛠️ Run the app:
   ```bash
   python webcam.py
   ```


## 📱 Contribute (Android App)
I am working on a Kivy-based Android app that brings the same Honey detection to mobile devices! 📲

**🎯 The mobile app:**
* Uses the device’s camera to detect Honey in real time.
* Plays a sound when Honey is detected.
* Has a Settings screen to switch cameras or toggle sound.
⚠️ The app is functional but not yet fully polished or packaged as an APK.
**🤝 I need help!**
* If you're experienced with Kivy, **Buildozer**, or packaging Android apps from Python...
* Or you’d like to help clean up the UI/UX...
**👉 Feel free to contribute or open a PR!**

## 🚧 To Do
* [x] 🌐 Convert the Python-based system into a Flask web app
* [x] 📱 Explore mobile/web compatibility
* [x] 🪟 Optional: Build a desktop version using Tkinter or PyQt
* [x] 🔒 Model security improvements
* [x] 🧪 Add unit tests and improve dataset validation

<<<<<<< HEAD
---

Made with ❤️ for a dog named Honey.
=======
Made with ❤️ for a dog named Honey.
>>>>>>> 7f99e2c0a8113c1e58ac744a52244d2dfb8c1365
