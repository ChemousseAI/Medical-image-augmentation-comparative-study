# Medical Image Augmentation via Generative Models

### A Comparative Study of Traditional Augmentation, DCGAN, and Diffusion Models for Diabetic Retinopathy Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

🏥 Medical Imaging | 🧠 Deep Learning | 🎨 Diffusion Models

---

## Overview

Deep learning has transformed medical image analysis, enabling automated disease diagnosis with performance approaching expert-level decision making in several clinical domains. However, the success of these systems strongly depends on the availability of large, diverse, and balanced annotated datasets.

In medical imaging, this requirement is rarely satisfied. Clinical datasets are often limited in size, severely imbalanced across pathological classes, expensive to annotate, and constrained by privacy regulations. These limitations significantly reduce the generalization capability of deep neural networks and frequently lead to overfitting.

This project investigates whether **generative deep learning can improve medical image classification by synthesizing realistic training samples**.

We perform a comparative study of three augmentation paradigms:

* Traditional image augmentation
* Deep Convolutional Generative Adversarial Networks (DCGAN)
* Diffusion-based image generation (DALL·E 2)

using retinal fundus images from the **IDRiD diabetic retinopathy dataset**.

The objective is to evaluate whether modern generative models can improve downstream classification performance under limited-data medical imaging conditions.

---

## Research Problem

Medical image classification can be formulated as a supervised learning problem.

Given a dataset:

$$D = {(x, y)}$$

where:

* $x$ = retinal fundus image
* $y ∈ {0,1,2,3,4}$ = diabetic retinopathy severity grade

we seek to learn a classifier:

$$f(x;θ) → y$$

by minimizing classification loss:

$$L(θ) = CrossEntropy(f(x), y)$$

However, limited dataset size causes poor estimation of the true data distribution:

$$P_train(x) ≠ P_real(x)$$

This leads to:

* poor generalization
* unstable optimization
* network overfitting
* weak performance on unseen data

The central research question is:

**Can synthetic image generation improve estimation of the underlying data distribution and improve medical image classification performance?**

---

## Dataset

This study uses the **Indian Diabetic Retinopathy Image Dataset (IDRiD)** [https://idrid.grand-challenge.org/].

Dataset characteristics:

* 516 retinal fundus images
* 413 training images
* 103 testing images
* 5 disease severity grades (0–4)

<p align="center">
  <img width="480" height="371" alt="image" src="https://github.com/user-attachments/assets/610367ae-e945-4f4d-a095-04f8ee9e380c" />
</p>

Challenges:

* highly limited sample size
* strong class imbalance
* expensive expert annotation

These characteristics make the dataset an ideal benchmark for augmentation research.

---

## Literature Review

Data augmentation has emerged as one of the most effective approaches for improving deep learning performance under limited-data conditions.

### Traditional Augmentation

Traditional augmentation applies deterministic transformations such as:

* flipping
* rotation
* cropping
* translation
* color shifting
* noise injection

These methods increase dataset size but do not introduce novel semantic information.



### GAN-Based Augmentation

Generative Adversarial Networks learn data distributions through adversarial optimization between two competing neural networks.

Advantages:

* realistic image synthesis
* unsupervised representation learning

Limitations:

* mode collapse
* unstable training
* difficult optimization

GAN-based augmentation has shown promising results in medical imaging, although training stability remains a major challenge.

---

### Diffusion-Based Augmentation

Diffusion probabilistic models have recently emerged as state-of-the-art generative models for image synthesis. Unlike GANs, diffusion models learn image generation through iterative denoising.

Advantages:

* stable training dynamics
* superior image fidelity
* improved mode coverage
* higher structural realism

Recent large-scale diffusion models have demonstrated unprecedented image generation capability.

---

## Methodology

We compare three augmentation strategies.

### 1. Traditional Data Augmentation

Classical image transformations were applied to increase dataset diversity.

* Geometric transformations:

$$x' = rotate(x, θ)$$

$$x' = translate(x, Δx, Δy)$$

$$x' = scale(x, s)$$

* Photometric transformations:

$$I' = αI + β$$

* Noise injection:

$$x' = x + ε$$

where ε follows Gaussian noise distribution.

Although these transformations increase sample count, they do not generate novel pathological structures.

---

### 2. DCGAN-Based Augmentation

DCGAN learns the data distribution through adversarial learning. The architecture contains:

* Generator: $G(z)$

* Discriminator: $D(x)$

* The generator maps random latent vectors into synthetic images. 

* The discriminator learns to distinguish real images from generated images.

* Optimization follows an adversarial objective:

* Generator → fool discriminator

* Discriminator → detect synthetic images

* Training process: min Generator, max Discriminator


Expected advantages:

* generation of novel retinal images
* learning latent anatomical structure

---

### DCGAN Architecture

<p align="center">
<img width="778" height="386" alt="image" src="https://github.com/user-attachments/assets/4c0cd03f-ca6f-4e26-91e2-a4b0a95cc4a4" />
</p>
---

### 3. Diffusion-Based Augmentation

Diffusion models learn image generation through two sequential processes.

### Forward Process

Noise is gradually added:

$$x_t = √(1-β)x_(t-1) + √β ε$$

where:

* $β$ = noise schedule
* $ε$ = Gaussian noise

After sufficient iterations:

$$x_T ≈ random noise$$

---

### Reverse Process

The model learns to progressively remove noise.

$$x_(t-1) = denoise(x_t)$$

The neural network predicts injected noise:

$$ε_pred = model(x_t,t)$$

Training minimizes prediction error:

$$Loss = ||ε − ε_pred||²$$

Unlike GANs, diffusion models directly learn probabilistic structure and exhibit significantly more stable optimization.

---

## Diffusion Model Selection

Several diffusion models were evaluated.

Tested models:

* Stable Diffusion
* GLIDE
* DALL·E 2

Images were generated using clinically descriptive prompts corresponding to diabetic retinopathy grades. The prompt specified photographs of the retina, which is the layer of tissue at the back of the eye that its Retinopathy grade = 1/4 and its Risk of macular edema = 0 for medical purpose.
<p align="center">
  <img src="https://github.com/user-attachments/assets/155b0a44-3bf6-4e01-8a31-06f7cc8344f8" width="280" hspace="10"/>
  <img src="https://github.com/user-attachments/assets/541410e5-a271-42bb-96ae-846e6726a8a9" width="280" hspace="10"/>
  <img src="https://github.com/user-attachments/assets/3a5f34a0-ddea-4f51-9761-789dda41ea2d" width="280" hspace="10"/>
</p>
Among all evaluated models, *DALL·E 2 produced the highest visual realism and strongest anatomical consistency*. Therefore DALL·E 2 was selected for augmentation experiments.



## Classification Models

Two classifiers were evaluated.

### Custom CNN

A lightweight convolutional neural network was designed specifically for this study.

<p align="center">
<img width="567" height="635" alt="image" src="https://github.com/user-attachments/assets/01396671-cef8-4b79-a416-628e327e2400" />
</p>

### ResNet50 Transfer Learning

A pretrained ResNet50 model was fine-tuned.

Modifications:

* replaced fully connected classification layers
* Xavier weight initialization
* Adam optimizer
* Cross-Entropy loss

Transfer learning was used to compensate for limited dataset size.

---

## Experimental Pipeline

The dataset was augmented until all classes reached balanced distribution.

Final dataset size: 1773 images

Training preprocessing:

* Z-score normalization
* balanced class distribution
* identical training hyperparameters across experiments

Each classifier was trained separately using:

* traditional augmentation
* diffusion augmentation

DCGAN-generated images were excluded after quality evaluation.



## Why DCGAN Was Excluded

DCGAN Experiments and Limitations

We initially investigated DCGAN as a generative approach for synthetic retinal image augmentation.
While the model successfully captured low-level retinal structures and global image distributions, the generated samples failed to achieve the anatomical fidelity required for downstream classification tasks.

Qualitative Assessment

Several limitations were consistently observed during training and sampling:

Training instability leading to inconsistent convergence behavior
Poor vascular structure representation, particularly in fine retinal vessels
Presence of visual artifacts and texture distortions
Limited ability to generate pathological diversity across disease categories
Reduced preservation of clinically relevant retinal patterns

These shortcomings significantly affected sample quality and limited the usefulness of the generated data for supervised learning experiments.

Generated Samples Analysis
<table align="center"> <tr> <th align="center">Training Phase Outputs</th> <th align="center">Sampling Phase Outputs</th> </tr> <tr> <td align="center"> <img src="https://github.com/user-attachments/assets/59055bc4-de21-4c16-a77c-786a5d807269" width="450"/> </td> <td align="center"> <img src="https://github.com/user-attachments/assets/0bec94ff-a3e5-4ad3-84b3-8ccdbe53cef3" width="450"/> </td> </tr> <tr> <td align="center"><em>(a) Images generated during adversarial training</em></td> <td align="center"><em>(b) Final synthetic samples produced after training</em></td> </tr> </table>
Training Dynamics

The adversarial training process exhibited significant instability, reflected by oscillatory generator–discriminator behavior and inconsistent convergence patterns.

<p align="center"> <img width="700" src="https://github.com/user-attachments/assets/4a8dfb01-57b0-418c-9fe6-561244ec9888" /> <br> <em>Figure 3: DCGAN training progression illustrating unstable convergence behavior.</em> </p>
Experimental Decision

Based on qualitative inspection and training behavior analysis, generated samples from DCGAN were considered insufficiently realistic for medical classification tasks.

As a result:

DCGAN-generated samples were excluded from the final comparative evaluation pipeline.

Subsequent experiments focused on more advanced generative approaches, particularly diffusion-based models, due to their superior structural fidelity and improved anatomical consistency.


## Experimental Results

### CNN Performance

| Augmentation Method | Accuracy | Loss |
| ------------------- | -------- | ---- |
| Traditional         | 34.95%   | 5.57 |
| Diffusion           | 41.75%   | 3.32 |


Improvement: **+6.8% accuracy**

---

### ResNet50 Performance

| Augmentation Method | Accuracy | Loss |
| ------------------- | -------- | ---- |
| Traditional         | 41.74%   | 2.11 |
| Diffusion           | 53.40%   | 1.36 |


Improvement: **+11.66% accuracy**

---

## Comparative Analysis

The experiments reveal clear differences between augmentation paradigms.

Traditional augmentation increases sample count but preserves semantic structure.

Diffusion models generate entirely new synthetic retinal structures that better approximate the true underlying data distribution.

Diffusion-generated images provided:

* higher classification accuracy
* reduced loss
* improved generalization
* reduced overfitting

The strongest improvements were observed when combined with transfer learning.

---

## Discussion

Several observations emerge from the experiments.

### Traditional augmentation has limited information gain

Although transformations increase dataset size, they do not introduce fundamentally new medical structures.

---

### Diffusion models improve distribution estimation

Synthetic diffusion-generated images more closely approximate:

$$P_generated(x) ≈ P_real(x)$$

This improves generalization during training.

---

### GAN instability limits medical applicability

DCGAN struggled to preserve fine anatomical retinal structures.

Medical imaging requires structural precision that GAN training instability often fails to capture.

---

### Transfer learning amplifies augmentation benefit

The pretrained ResNet50 extracted stronger hierarchical visual representations.

This allowed diffusion-generated images to improve downstream classification more effectively.

---

## Conclusion

This work investigated three augmentation paradigms for diabetic retinopathy classification under limited-data medical imaging conditions.

Experimental evidence demonstrates:

* traditional augmentation improves robustness but introduces limited semantic diversity
* DCGAN failed to generate sufficiently realistic retinal images
* diffusion-based augmentation consistently outperformed traditional methods

Best performance: **ResNet50 + Diffusion Augmentation**

Accuracy: **53.40%**

These findings suggest that diffusion models represent a promising direction for synthetic medical image generation and data augmentation in clinical deep learning systems.

---

## References

[1] Goodfellow et al. Generative Adversarial Networks. NeurIPS 2014

[2] Radford et al. DCGAN. arXiv 2015

[3] Ho et al. Denoising Diffusion Probabilistic Models. NeurIPS 2020

[4] He et al. Deep Residual Learning for Image Recognition. CVPR 2016

[5] Porwal et al. IDRiD Dataset. IEEE Dataport 2018

[6] Shorten et al. Survey on Image Data Augmentation for Deep Learning. Journal of Big Data 2019
