## Expriment Records

### Versions

* ALPHA_VOC: 最初版本，DPL模型，训练了47个Epoch，lr=0.0002

* ALPHA_VOC_2: 利用 ALPHA_VOC 14 epoch 的参数， lr=0.0001 进行 pretrained, 模型结束前，mAP=0.28

* ALPHA_VOC_3: 优化所有参数，在 ALPHA_VOC_2 的参数基础之上训练

-----

**以上三个模型均没有用到VGG Pretrained Model**

* BETA_VOC: 完善DPL, 使用VGG pretrained model


|  ID |  Epoch  |  mAP |
|-----|---------|------|
|vgg_pretrained_5_epoch_20180523|5|xx|


* RESNET50_VOC: 使用ResNet50 pretrained model

|  ID |  Epoch  |  mAP |
|-----|---------|------|
|x|x|x|