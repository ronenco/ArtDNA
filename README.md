# 🧠 ArtDNA - AI Author Classifier

Identify which **AI model** (e.g. *Midjourney, DALL·E, Stable Diffusion*) most likely generated a given image — using visual fingerprinting, statistical cues, and style embeddings.

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

## 🤝 Contributors

Team project for the AI Author Classification initiative
Contributors:

- Cohen, Ronen
- Lowte, Oren
- Malikov, Mark
- Talmor, Oren

---
