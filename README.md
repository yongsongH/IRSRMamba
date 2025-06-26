# [IRSRMamba](http://arxiv.org/abs/2405.09873)
Official PyTorch implementation of the paper [IRSRMamba: Infrared Image Super-Resolution via Mamba-based Wavelet Transform Feature Modulation Model.](http://arxiv.org/abs/2405.09873)

#### 🚩Accepted by IEEE TGRS

*I would like to express our sincere gratitude to all the authors whose work has contributed to this project. Their insights and contributions have been invaluable. We are also deeply thankful for the positive feedback and constructive assistance received from the editors and reviewers. Their expertise and thoughtful suggestions have significantly improved the quality and clarity of this work.*

## Introduction

Infrared image super-resolution demands long-range dependency modeling and multi-scale feature extraction to address challenges such as homogeneous backgrounds, weak edges, and sparse textures. While Mamba-based state-space models (SSMs) excel in global dependency modeling with linear complexity, their block-wise processing disrupts spatial consistency, limiting their effectiveness for IR image reconstruction. We propose IRSRMamba, a novel framework integrating wavelet transform feature modulation for multi-scale adaptation and an SSMs-based semantic consistency loss to restore fragmented contextual information. This design enhances global-local feature fusion, structural coherence, and fine-detail preservation while mitigating block-induced artifacts. Experiments on benchmark datasets demonstrate that IRSRMamba outperforms state-of-the-art methods in PSNR, SSIM, and perceptual quality. This work establishes Mamba-based architectures as a promising direction for high-fidelity IR image enhancement.

## Approach overview

![IRSRMamba](experiments/pretrained_models/IRSRMamba.png)


Please check [here.](https://github.com/yongsongH/IRSRMamba/blob/3fb448b0efaa5ded1bd2b878d9535e256f99509f/experiments/pretrained_models/vis.pdf)


## Requirements
> - Python 3.8, PyTorch >= 1.11
> - BasicSR 1.4.2
> - Platforms: Ubuntu 18.04, cuda-11



## Installation
>  Clone the repo
```
git clone https://github.com/yongsongH/IRSRMamba.git
```
> Install dependent packages
```
cd IRSRMamba
```
```
pip install -r install.txt
```
> Install BasicSR
```
python setup.py develop
```
***You can also refer to this [INSTALL.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md) for installation***

## Dataset prepare

Please check this [page](https://figshare.com/articles/dataset/IRSRMamba_Infrared_Image_Super-Resolution_via_Mamba-based_Wavelet_Transform_Feature_Modulation_Model/25835938).

## Model

Pre-trained models can be downloaded from this [link](https://figshare.com/articles/dataset/IRSRMamba_Infrared_Image_Super-Resolution_via_Mamba-based_Wavelet_Transform_Feature_Modulation_Model/25835938).

## Evaluation

please check the [log file](https://github.com/yongsongH/IRSRMamba/blob/main/results/0515_SPL_IRSRMamba_Final_x2/test_0515_SPL_IRSRMamba_Final_x2_20240516_171818.log) for more information about the settings.

### Training
- Run the following commands for training:
```
python basicsr/train.py -opt options/train/train_IRSRMamba_final_x2.yml
```
```
python basicsr/train.py -opt options/train/train_IRSRMamba_final_x4.yml
```
    
### Testing
- Download the pretrained models.
- Download the testing dataset.
- Run the following commands:
```
  python basicsr/test.py -opt options/test/test_IRSRMamba_x4.yml
```
```
  python basicsr/test.py -opt options/test/test_IRSRMamba_x2.yml
```

## Contact

If you meet any problems, please describe them and [contact](https://hyongsong.work/) me. 

**Impolite or anonymous emails are not welcome. There may be some difficulties for me to respond to the email without self-introduce. Thank you for understanding.**

## Acknowledgement
This work is under peer review.
The updated manuscript and training dataset will be released after the paper is accepted. 


---

## Legal Action and Patent Clause
This project is licensed under the **Apache License, Version 2.0**.
A copy of the Apache License, Version 2.0, is included in the `LICENSE` file in this repository. You may also obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

**Any violation of the terms of the Apache License, Version 2.0, may result in legal action and liability for damages.**
