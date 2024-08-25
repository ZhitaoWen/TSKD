The official implementation of ` A Triple Semantic-aware Knowledge Distillation Network for Industrial Defect Detection`.


![struture](assets/framewor.png)

## Install
conda create --name tskd python=3.7 -y
conda activate tskd
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install "mmengine==0.7.3"
mim install "mmcv==2.0.0rc4"
git clone https://github.com/Zhitaowen/TSKD
cd TSKD
pip install -v -e .

## dataset

Prepare dataset follow the [official instructions](https://mmdetection.readthedocs.io/en/3.x/user_guides/dataset_prepare.html).



### Training

```shell
python tools/train.py configs/tskd/${CONFIG_FILE} [optional arguments]
```

### Evaluation

```shell
python tools/test.py configs/tskd/${CONFIG_FILE} ${CHECKPOINT_FILE}



