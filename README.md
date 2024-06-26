# Latent Factor learning by Data Augmentation (LFDA)

Implementation for intracortical brain-computer interfaces for long-term use.  
Feel free to utilize the code and cite this paper.  
Shih-Hung Yang, Chun-Jui Huang, and Jhih-Siang Huang. “Increasing Robustness of Intracortical Brain-Computer Interfaces for Recording Condition Changes via Data Augmentation.” Computer Methods and Programs in Biomedicine, 2024.  
https://doi.org/10.1016/j.cmpb.2024.108208

## ABSTRACT
**Background and Objective:** Intracortical brain-computer interfaces (iBCIs) aim to help paralyzed individuals restore their motor functions by decoding neural activity into intended movement. However, changes in neural recording conditions hinder the decoding performance of iBCIs, mainly because the neural-to-kinematic mappings shift. Conventional approaches involve either training the neural decoders using large datasets before deploying the iBCI or conducting frequent calibrations during its operation. However, collecting data for extended periods can cause user fatigue, negatively impacting the quality and consistency of neural signals. Furthermore, frequent calibration imposes a substantial computational load.

**Methods:** This study proposes a novel approach to increase iBCIs’ robustness against changing recording conditions. The approach uses three neural augmentation operators to generate augmented neural activity that mimics common recording conditions. Then, contrastive learning is used to learn latent factors by maximizing the similarity between the augmented neural activities. The learned factors are expected to remain stable despite varying recording conditions and maintain a consistent correlation with the intended movement.

**Results:** Experimental results demonstrate that the proposed iBCI outperformed the state27 of-the-art iBCIs and was robust to changing recording conditions across days for long-term use on one publicly available nonhuman primate dataset. It achieved satisfactory offline decoding performance, even when a large training dataset was unavailable.

**Conclusions:** This study paves the way for reducing the need for frequent calibration of iBCIs and collecting a large amount of annotated training data. Potential future works aim to improve offline decoding performance with an ultra-small training dataset and improve the iBCIs’ robustness to severely disabled electrodes.
