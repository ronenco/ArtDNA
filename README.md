# 🧠 ArtDNA - AI Author Classifier

Identify which **AI model** (*Midjourney vs DALL·E*) most likely generated a given image — using visual fingerprinting, statistical cues, and style embeddings.

---

## 🎯 Project Overview

As AI-generated images become increasingly realistic, distinguishing between their sources has become a complex and relevant challenge.  
Different generative models — such as *Midjourney*, *DALL·E*, *Stable Diffusion*, or *Ideogram* — produce images that share visual realism but differ subtly in composition, texture, and statistical “style signatures.”

This project explores **automated author attribution** — determining which AI model produced a given image — by leveraging **transfer learning** and **architecture comparison** techniques.

### 🧠 Methodology & Architecture

Instead of training models entirely from scratch, we utilize **transfer learning** from well-established computer vision architectures (e.g., **ResNet50**, **VGG16**, **EfficientNet**, **Vision Transformers**) that were pretrained on large-scale datasets such as ImageNet.  
These networks already capture general visual features like color distribution, edge patterns, object composition, and texture statistics — all of which can serve as distinctive fingerprints for generative models.

By fine-tuning only the final classification layers, we adapt these pretrained networks to our specific task: **classifying the source model** of a given image.  
This approach:
- 🧩 **Reduces training time and resource requirements**  
- 🎯 **Improves accuracy with limited datasets**  
- 🔬 **Highlights transferable features** between natural and AI-generated imagery  

### ⚙️ Comparative Evaluation

To better understand the underlying characteristics of different architectures, we compare multiple feature extractors and classifier heads:
- **CNN-based models** (e.g., ResNet, VGG) capture localized spatial features and fine textural differences.
- **Transformer-based models** (e.g., ViT, Swin) analyze global relationships and compositional coherence.
- Hybrid or ensemble methods combine the strengths of both.

Through systematic experiments, we evaluate how well each architecture separates different AI sources and whether certain architectures are more sensitive to particular stylistic traits (e.g., brushstroke-like noise, color palette bias, or text rendering quality).

### 🧩 Objective

The goal is to build a **reproducible framework** for AI model attribution — one that:
- Benchmarks multiple architectures using transfer learning
- Identifies key visual features correlated with each model
- Provides insights for **academic research**, **digital forensics**, and **content authenticity verification**

Ultimately, this project aims to contribute to the broader effort of **AI transparency and accountability** by revealing the distinct visual fingerprints of generative models.

---

## 🧩 Repository Structure

```
├── data/         # datasets or links to datasets (excluded from git if large)
├── notebooks/    # exploratory analysis, training, visualizations
├── src/          # reusable source code: loaders, features, classifiers
├── docs/         # documentation, diagrams, research notes
└── README.md
```
---

## 📦 Installation

This project supports all operating systems (Windows, macOS, Linux), with an optional macOS-optimized TensorFlow setup that uses the Metal GPU backend for faster training on Apple Silicon (M1/M2/M3).

Follow the steps below to set up your environment.

### 1️⃣ Clone the Repository

```
git https://github.com/ronenco/ArtDNA.git
cd ArtDNA
```

### 2️⃣ Create a Virtual Environment (recommended)

We assume you have python3 installed.

```
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux

# or on Windows (PowerShell):
# .venv\Scripts\activate
```

### 3️⃣ Install Dependencies

#### ✔ Option A — OS-Agnostic Installation (recommended for shared environments)

Use this if you want your environment to behave the same across Windows, Linux, and macOS:

```
pip install -r requirements.txt
```

This installs:
	•	TensorFlow (CPU version)
	•	NumPy
	•	Matplotlib
	•	scikit-learn
	•	Pillow
	•	seaborn
	•	tqdm
…and all other required packages.

#### ✔ Option B — macOS (Apple Silicon) Accelerated Installation

Use this if your machine is an M1/M2/M3 Mac and you want GPU acceleration via Metal.

Install the CPU-agnostic packages first:

```
pip install -r requirements.txt
```

Then install the Apple-specific TensorFlow backend:

```
pip install -r requirements-mac.txt
```

This installs:
- tensorflow-macos
- tensorflow-metal

which replaces the standard TensorFlow wheel with a GPU-optimized implementation.

[!NOTE]
⚠️ Note: Do NOT install both tensorflow and tensorflow-macos together inside the same requirements file — installing them sequentially (as above) is the safe method. The macOS packages will automatically override the CPU version.

### 4️⃣ Verify Installation

Run:

```
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

On macOS with Metal enabled, you should also confirm your GPU is visible:

```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output (mac only):

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 🤝 Contributors

Team project for the AI Author Classification initiative
Contributors:

- Cohen, Ronen
- Lowte, Oren
- Malikov, Mark
- Talmor, Alon

---
