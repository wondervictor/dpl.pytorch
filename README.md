# dpl.pytorch
Pytorch Implementation for "Deep Patch Learning for Weakly Supervised Object Classification and Discovery" [paper](https://arxiv.org/abs/1705.02429)

## Results

**PASCAL VOC2012 Testset**

**mAP: 0.90240**

|Class|aeroplane|bicycle|bird   |boat   |bottle |bus    |car    |cat    |chair  |cow    |
|-----|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|AP   |0.98400  |0.92760|0.95630|0.93490|0.77990|0.92200|0.90910|0.97970|0.81800|0.90490|
|Class|diningtable|dog    |horse |motorbike|person |pottedplant|sheep  |sofa   |train  |tvmonitor|
|AP   |0.79660    |0.97180|0.96420|0.94030   |0.97750|0.70770    |0.92720|0.77180|0.97240|0.90220|

## Training

* Compling libs for this framework

```bash

cd lib/model/
cd roi_align/ && ./make.sh
cd roi_pooling && ./make.sh
cd spmmax_pooling && ./make.sh

```

* train

```bash
python train.py --imageset [train, trainval] --basemodel [vgg, resnet34, resnet50] --data_dir <Data Directory Path>
```

* proposal
    - densebox sampling
    - selective search
 
## SPMMAX Pooling For **PyTorch**

written as a PyTorch Extension and supported CUDA

see: `./lib/model/spmmax_pooling`


## Licence

**This project is under MIT Licence**