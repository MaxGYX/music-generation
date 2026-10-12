# Music Generation via Image Models (VAE + Diffusion)

## 📖 Overview
This project explores an innovative approach to AI music generation. Music can be represented in various forms, such as instrument playing instructions (MIDI) or raw sound wave superpositions. Instead of these traditional methods, this project converts music into **spectrograms (images)** and utilizes advanced image generation models, specifically **VAE (Variational Autoencoder)** and **Diffusion models**, to generate new musical compositions.
<div align="left">
  <img width="598" height="269" alt="image" src="https://github.com/user-attachments/assets/381c74f8-2363-4253-9ff3-8907f2b9f18a" />
</div>

## ✨ Motivation & Innovation
- **Visualizing Sound:** While MIDI-based generation is very common, using state-of-the-art image generation models to create music is a relatively new frontier, inspired by recent research from Google.
- **Curiosity-Driven:** I was deeply impressed by the performance of image generation models and wanted to understand their underlying principles through hands-on practice. I was also curious to see what music generated from image models would sound like and whether it could maintain rhythmic and melodic continuity.

## 🧗 Challenges & Reflections
During the research process, the mathematical principles behind VAE and Diffusion were initially quite difficult to grasp. However, with the guidance of my AI research mentors and repeated study, I gradually understood them.

The initial generated music did not fully meet my expectations. Through analysis, I identified the following key reasons:
1. **Feature Mismatch:** Existing image generation models (like VAE/Diffusion) are not specifically trained for music spectrograms. They extract features based on physical objects in typical photos, lacking specialized designs for musical elements like melody continuity and rhythm.
2. **Standard Architectures:** Standard models are not tailored to interpret the unique spatial representation of audio frequencies and time on a spectrogram.

## 🏗️ Architecture
The following diagram illustrates the complete pipeline of converting audio to spectrograms, applying VAE and Diffusion models in the latent space, and reconstructing the audio.
<div align="center">
  <img width="993" height="656" alt="image" src="https://github.com/user-attachments/assets/67e7c25f-8c18-472c-92b9-34ad4e4faa1a" />
</div>

## 🚀 Future Work
To improve the generation quality, I plan to explore the following directions in the future:
- Curating a specialized dataset of music spectrograms (e.g., focusing on a specific genre of music).
- Designing specialized convolution kernels to better extract melodic and rhythmic features from spectrograms.
- Retraining or fine-tuning models specifically for music generation based on these specialized features.

## 🧠 Key Learnings
Through this project, I have achieved significant personal growth:
- Gained a deep understanding of the principles of Deep Neural Networks.
- Mastered the structure and principles of various architectures, including **VAE, Diffusion, CNN, RNN, and Transformer**.
- Built a strong belief in the vast future possibilities unlocked by AI development.
