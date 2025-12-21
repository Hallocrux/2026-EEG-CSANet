<div align="center">

# EEG-CSANet

## Fusion of Multiscale Features via Centralized Sparse-Attention Network for EEG Decoding

👥 **Authors**: Xiangrui Cai<sup>1</sup>, Shaocheng Ma<sup>1</sup>, Lei Cao*, Jie Li*, Tianyu Liu*, and Yiling Dong.  

🏫 **Affiliations**: Shanghai Maritime University & Tongji University  

[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/XXXX.XXXXX)

</div>

## Model
![Modle](/Figure/Model.png)

- We propose a novel Fusion of Multiscale Features via Centralized Sparse-attention Network (EEG-CSANet), which effectively addresses the loss of channel discriminative information caused by coarse-grained feature fusion in the traditional multiscale temporal feature integration process.
- We propose a feature fusion architecture that synergistically integrates a primary branch with an auxiliary branch. The primary branch employs a multi-scale multi-head self-attention
mechanism to enhance the modeling of core spatiotemporal patterns, while the auxiliary branch leverages a multi-scale sparse multi-head cross-attention mechanism to enable efficient and precise feature interactions with the local key regions of the primary branch.
- We conduct experiments on five public datasets (BCIC-IV-2a, BCIC-IV-2b, HGD, SEED, and SEED-VIG), and the results demonstrate that our method consistently achieves superior classification accuracy and generalization performance compared to existing approaches, thereby validating its effectiveness and robustness.

## Datasets
- **[BCIC-IV-2A](https://www.bbci.de/competition/iv/)**  
- **[BCIC-IV-2B](https://www.bbci.de/competition/iv/)**  
- **[HGD](https://gin.g-node.org/robintibor/high-gamma-dataset)**  
- **[SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html)**  
- **[SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html)**

## Results
<div align="center">
  
![Result_1](/Figure/Result_1.png)
![Result_2](/Figure/Result_2.png)
![Result_3](/Figure/Result_3.png)
![Result_4](/Figure/Result_4.png)

</div>

## Acknowledge
We would like to express our sincere gratitude to the following open-source projects that have contributed to this work:
- https://github.com/Ma-Xinzhi/EEG-TransNet
- https://github.com/eeyhsong/EEG-Conformer

## References
This paper is currently under submission.

If you have any questions regarding our manuscript or the associated code, please feel free to contact us at:  
- 202330310112@stu.shmtu.edu.cn  
- 202430310086@stu.shmtu.edu.cn
