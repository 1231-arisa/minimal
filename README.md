---
title: AIstylist
emoji: 👕
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app_flask.py
pinned: false
---

# 🧠 AIstylist Proof of Concept

A Python tool that automatically trims clothing images and composites them onto avatar illustrations. This project aims to create a simple yet effective way to visualize clothing items on virtual avatars.

## ✨ Features

* Automatic clothing image trimming using advanced AI segmentation
* Avatar-clothing compositing with intelligent positioning
* Modern web interface (Flask + HTML/CSS/JS)
* Drag-and-drop file upload
* Real-time image previews
* Organized project structure for easy development

## 🗂️ Project Structure

```
app_flask.py        # Flask web application
minimal.py          # Core logic for image processing
templates/          # HTML templates
  └── index.html    # Main web interface
input/              # Place your avatar and clothing images here
output/             # Composite image output
results/            # Save logs or future outputs here
data/               # Optional input data (not used in current script)
```

## 📋 Requirements

* Python 3.7 or higher
* Flask web framework
* OpenCV for image processing
* rembg for AI-powered background removal
* Pillow for image manipulation

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/1231-arisa/minimal.git
cd minimal
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Web Interface (Recommended)
1. Run the Flask application:
```bash
python app_flask.py
```

2. Open your browser and go to `http://localhost:7860`

3. Upload your avatar and clothing images using the web interface

4. Click "Process Images" to generate the composite

### Command Line Interface
1. Place your images in the `input/` folder:  
   * Avatar image (e.g., `avatar.png`)  
   * Clothing image (e.g., `Navyshirt.jpg`)
2. Run the script:
```bash
python minimal.py
```
3. Check the `output/` folder for the composite image

## 🔮 Future Plans

* Add support for multiple clothing items
* Implement automatic background removal
* Add GUI interface
* Support for different avatar poses
* Batch processing capability
* Real-time video processing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
