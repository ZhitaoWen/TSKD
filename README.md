The official implementation of ` A Triple Semantic-aware Knowledge Distillation Network for Industrial Defect Detection`.


![struture](assets/framework.png)

## Introduction
Knowledge distillation (KD) is a powerful model compression technique that aims to transfer knowledge
from heavy teacher networks to compact student networks via distillation. However, effectively transferring
semantic knowledge in industrial settings poses significant challenges. On one hand, the appearance of defects
(e.g., size and shape) may vary considerably due to the influence of the industrial site, which potentially
weakens the semantic associations between class-specific features. On the other hand, agnostic background
interference (e.g., spike anomalies and low light) may foster semantic ambiguity of class-specific features. As
such, the weakened semantic associations and fostered semantic ambiguities hinder the efficacy and adequacy
of knowledge transfer in KD. To mitigate these limitations, we propose a triple semantic-aware knowledge
distillation (TSKD) network for industrial defect detection. TSKD contains three refinements, i.e., dual-relation
distillation (DRD), decoupled expert distillation (DED), and cross-response distillation (CRD). Specifically, DRD
employs graph reasoning networks to strengthen semantic associations at both the instance and pixel levels,
DED enhances semantic explicitness by decoupling foreground and background features while injecting expert
priors, and CRD further captures task-specific semantic response knowledge. By integrating these components,
TSKD can effectively perceive triple semantic knowledge of relations, features, and responses, ensuring more
robust and comprehensive knowledge transfer. Experimental evaluations on two challenging industrial datasets
show that TSKD can significantly improve detector performance (MFL-DET: 98.9% mAP; NEU-DET: 81.0%
mAP) and compress computation (MFL-DET: 19.7M Params and 105 FPS; NEU-DET: 19.7M Params and 116
FPS).

## Install
```shell
conda create --name tskd python=3.7 -y
conda activate tskd
- conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
- pip install -U openmim
- mim install "mmengine==0.7.3"
- mim install "mmcv==2.0.0rc4"
- git clone https://github.com/Zhitaowen/TSKD
- cd TSKD
- pip install -v -e .
```

## dataset

Prepare dataset follow the [official instructions](https://mmdetection.readthedocs.io/en/3.x/user_guides/dataset_prepare.html).



## Training

```shell
python tools/train.py configs/tskd/${CONFIG_FILE} [optional arguments]
```

## Evaluation

```shell
python tools/test.py configs/tskd/${CONFIG_FILE} ${CHECKPOINT_FILE}
```

## Citation

If you find our repo useful for your research, please cite us:

```latex
@article{wen2025triple,
  title={A triple semantic-aware knowledge distillation network for industrial defect detection},
  author={Wen, Zhitao and Liu, Jinhai and Zhao, He and Wang, Qiannan},
  journal={Computers in Industry},
  volume={166},
  pages={104252},
  year={2025},
  publisher={Elsevier}
}
```
## Acknowledgement
We sincerely thank  [mmdetection](https://github.com/open-mmlab/mmdetection), [FGD](https://github.com/yzd-v/FGD), and [Cross-KD](https://github.com/jbwang1997/CrossKD) for providing their wonderful code to the community!


