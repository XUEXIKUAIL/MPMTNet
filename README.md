## Multiattentive Perception and Multilayer Transfer Network Using Knowledge Distillation for RGB-D Indoor Scene Parsing（MPMTNet-KD）

[![Powered by](https://camo.githubusercontent.com/cb046b12378f0b5471f89a372a423b9d6b14c62a8632c82d748a34293da9185b/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f42617365645f6f6e2d5079746f7263682d626c75653f6c6f676f3d7079746f726368)](https://camo.githubusercontent.com/cb046b12378f0b5471f89a372a423b9d6b14c62a8632c82d748a34293da9185b/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f42617365645f6f6e2d5079746f7263682d626c75653f6c6f676f3d7079746f726368)  [![Ask Me Anything!](https://camo.githubusercontent.com/df82a0ad8bd16fc23282356195f14b5b2e78a5e146d1e4881270d040e9207853/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4f6666696369616c2532302d5965732d3161626339632e737667)]([XUEXIKUAIL/MPMTNet](https://github.com/XUEXIKUAIL/MPMTNet))  [![Ask Me Anything!](https://img.shields.io/github/last-commit/XUEXIKUAIL/MPMTNet.svg)]([XUEXIKUAIL/MPMTNet](https://github.com/XUEXIKUAIL/MPMTNet))

🔮 Welcome to the official code repository for [our paper(TBD)](https://ieeexplore.ieee.org). We are excited to share our work with you, so please bear with us as we prepare the detailed code. 

🔮 Stay tuned for the reveal, and please [contact us](a133021478@163.com) via email if you need help!

### Illustration of Idea

💡 Scene parsing has gained wide attention in the field of computer vision, with emerging methods and techniques providing superior solutions. Although some of the methods improve performance, they tend to neglect the number of model parameters and computational size. To address these limitations, we propose the **multiattentive perception and multilayer transfer network that employs knowledge distillation (MPMTNet-KD)**.

### Framework

![image-20250408171223382](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250408171223382.png)

### Implementation

#### 💻Requirements

The code has been tested and verified using Python >=3.6, Pytorch >= 1.7.1 and Cuda = 10.2. However, compatibility with other versions is also likely.

If you already have a good Conda environment, you can install dependencies to compensate for missing packages as follows:

```
pip install -r ./requirement.txt
```

Otherwise you may need to create Conda Environment and Install Dependencies:

```
# create new conda env
conda create -n mpmtnet python=3.7 -y
conda activate mpmtnet

# install python dependencies
pip install -e .
# [optional] install python dependencies for you!
pip install -r ./requirements.txt
```

#### ⚡Dataset Preparation

NYUDv2 dataset can be download here [NYUDv2](https://drive.google.com/drive/folders/1tief3fgaTe2hown8FRnrb9ZtsMeoWtlv). # You need to change the data root in ./configs/[nyuv2.json](https://github.com/XUEXIKUAIL/EAMINet/blob/main/configs/nyuv2.json)

SUN-RGBD dataset can be download here [SUN-RGBD](https://rgbd.cs.princeton.edu/data/SUNRGBD.zip). # You need to change the data root in ./configs/[sunrgbd.json](https://github.com/XUEXIKUAIL/EAMINet/blob/main/configs/sunrgbd.json)

#### 🔔 For Training

1. Before model training begins, we use the [Segformer family(Paper)](https://arxiv.org/abs/2105.15203) of models as our backbone and the pre-trained weights are initialized. You can obtain them from：[Segformer pre-trained weight(Downloads)](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw).

2. Put the segformer pre-trained weight in the following file ./train.py, and refer to the modification as shown.

######      ![image-20250408164118359](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250408164118359.png) 

3. Now，You can run train.py！

#### 🔔 For Evaluate

If you only want to test the performance, we also provide our trained weights for results reproducibility.


| <span style="display:inline-block;width:80px">Dataset</span> | <span style="display:inline-block;width:100px">Pretrained_Model_url</span> | <span style="display:inline-block;width:60px">mIoU</span> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-------------------------------------------------------: |
|                            NYUDv2                            | [Baidu-Pan](https://pan.baidu.com/s/1KtX4jIwsQi_-ZgbrDxPEWQ?pwd=ghp3) |                        54.9(1.7↑)                         |
|                           SUN-RGBD                           | [Baidu-Pan](https://pan.baidu.com/s/1KtX4jIwsQi_-ZgbrDxPEWQ?pwd=ghp3) |                        50.8(1.6↑)                         |

Put our pre-trained weight in the following file ./evaluate.py, and run [evaluate.py]([MPMTNet/evaluate.py at main · XUEXIKUAIL/MPMTNet](https://github.com/XUEXIKUAIL/MPMTNet/blob/main/evaluate.py))！

 ![image-20250408171103248](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250408171103248.png)

### Experiments

📊Our paper achieves excellent performance improvement on challenging tasks such as RGB-D Indoor Scene Parsing and RGB-T Scene Parsing. Additional detailed results are presented in [our paper(TBD)](https://ieeexplore.ieee.org).

​						**Table I**   Evaluation metrics on NYUv2 obtained from compared methods. The best results are shown in bold.

| Model           | Published Year | Backbone      | NYUv2 |        |         |           |          |       |
| --------------- | -------------- | ------------- | ----- | ------ | ------- | --------- | -------- | ----- |
|                 |                |               | PA(%) | CMA(%) | mIoU(%) | Params(M) | FLOPs(G) | FPS ↑ |
| RDFNet [12]     | ICCV2017       | ResNet101     | 76.0  | 62.8   | 50.1    | 443.0     | 1.3T     | 11.4  |
| ACNet [17]      | ICIP2019       | ResNet50      | -     | -      | 48.3    | 116.6     | 250.1    | 21.2  |
| ESANet [18]     | ICRA2021       | ResNet34      | -     | -      | 51.5    | 44.1      | 47.9     | 42.6  |
| SA-Gate [14]    | ECCV2020       | ResNet50      | 77.9  | -      | 52.4    | 110.6     | 359.8    | 16.6  |
| MGCNet [35]     | SPL2022        | ConvNeXt-S    | 78.7  | 68.3   | 54.5    | 208.6     | 230.2    | 10.1  |
| FRNet [31]      | JSTSP2022      | ResNet50      | 77.6  | 66.5   | 53.6    | 87.8      | 231.1    | 25.8  |
| PGDENet[28]     | TMM2023        | ResNet34      | 78.1  | 66.7   | 53.7    | 107.4     | 357.5    | 17.2  |
| SGACNet[36]     | IEEE SJ2023    | ResNet101     | 76. 8 | 63.3   | 51.1    | 58.3      | 69.4     | 15.0  |
| PDCNet [29]     | TCSVT2023      | ResNet101     | 78.4  | -      | 53.5    | -         | -        | -     |
| BCINet [32]     | IF2023         | ResNet50      | 77.1  | 66.7   | 52.9    | 78.3      | 97.3     | 27.3  |
| CMXNet [37]     | TITS2023       | SegformerB2   | 78.7  | -      | 54.1    | 66.6      | 133.6    | 14.8  |
| EFDCNet [38]    | IMAVIS2024     | ResNet50      | 77.4  | 65.4   | 51.9    | 72.5      | 66.1     | -     |
| SwinNet[39]     | JVCI2024       | Swin-Large    | 77.5  | 65.8   | 52.4    | -         | -        | -     |
| CCANet [40]     | TCDS2024       | SegformerB2   | 78.5  | 66.7   | 53.4    | -         | -        | -     |
| AsymFormer [33] | CVPR2024       | SegformerB0+T | 77.0  | -      | 54.1    | 33.0      | 36.0     | -     |
| MPMTNet-T       |                | SegformerB4   | 79.9  | 73.0   | 57.7    | 125.8     | 129.4    | 9.9   |
| MPMTNet-S       |                | SegformerB2   | 77.7  | 68.2   | 53.2    | 28.3      | 62.2     | 17.7  |
| MPMTNet-KD      |                | SegformerB2   | 78.5  | 69.3   | 54.9    | 28.3      | 62.2     | 17.7  |

​					**Table II**   Evaluation metrics on SUN RGB-D obtained from compared methods. The best results are shown in bold.

| Model           | Published Year | SUN RGB-D |      |      |
| --------------- | -------------- | --------- | ---- | ---- |
|                 |                | PA        | CMA  | mIoU |
| RDFNet [12]     | ICCV2017       | 81.5      | 60.1 | 47.7 |
| ACNet [17]      | ICIP2019       | -         | -    | 48.1 |
| ESANet [18]     | ICRA2021       | -         | -    | 48.3 |
| SA-Gate [14]    | ECCV2020       | 82.5      | -    | 49.4 |
| MGCNet [35]     | SPL2022        | 86.5      | 64.2 | 51.5 |
| FRNet [31]      | JSTSP2022      | 87.4      | 62.2 | 51.8 |
| PGDENet [28]    | TMM2023        | 87.7      | 61.7 | 51.0 |
| SGACNet [36]    | IEEE SJ2023    | 81.8      | 60.9 | 48.5 |
| PDCNet [29]     | TCSVT2023      | 82.4      | -    | 49.2 |
| BCINet [32]     | IF2023         | 82.3      | -    | 49.2 |
| CMXNet [37]     | IMAVIS2024     | 82.8      | -    | 49.7 |
| EFDCNet [38]    | DSP2024        | 82.6      | 61.5 | 49.2 |
| SwinNet [39]    | JVCI2024       | 80.2      | 59.2 | 47.3 |
| CCANet [40]     | TCDS2024       | 82.3      | 59.8 | 48.6 |
| AsymFormer [33] | CVPR2024       | 81.9      | -    | 49.1 |
| MPMTNet-T       |                | 82.5      | 65.9 | 51.7 |
| MPMTNet-S       |                | 81.7      | 63.3 | 49.2 |
| MPMTNet-KD      |                | 82.2      | 64.1 | 50.8 |
 
   ​**Table III**   Evaluation metrics on MFNet obtained from compared methods. The best results are shown in bold.

| Model                       | Backbone    | MFNet Dataset |            |
| --------------------------- | ----------- | ------------- | ---------- |
|                             |             | mAcc(%)       | mIoU(%)    |
| MFNet <sub>17</sub>[60]     | -           | 45.1          | 39.7       |
| ACNet <sub>19</sub>[17]     | ResNet-50   | -             | 46.3       |
| RTFNet <sub>19</sub>[62]    | ResNet-152  | 63.1          | 53.2       |
| PSTNet <sub>20</sub>[61]    | ResNet-18   | -             | 48.4       |
| ABMDRNet <sub>21</sub>[63]  | ResNet-50   | 69.5          | 54.8       |
| MMDRNet <sub>23</sub>[64]   | ResNet-50   | 72.4          | 56.0       |
| HAFFSeg <sub>23</sub>[65]   | MobileViT   | 73.8          | 59.2       |
| FDCNet <sub>23</sub>[66]    | ResNet-50   | 74.1          | 56.3       |
| MMSMCNet <sub>23</sub>[67]  | SegformerB3 | 75.2          | 58.1       |
| CLSNet-S* <sub>24</sub>[68] | SegformerB2 | 74.7          | 57.3       |
| MMDNet <sub>24</sub>[69]    | ResNet-50   | 74.7          | 56.8       |
| MDBFNet <sub>24</sub>[70]   | ResNet-50   | 76.6          | 55.6       |
| RSFNet <sub>24</sub>[71]    | ResNet101   | 73.6          | 55.1       |
| MPMTNet-T                   | SegformerB4 | 74.1          | 54.7       |
| MPMTNet-S                   | SegformerB2 | 70.3          | 50.8       |
| MPMTNet-KD                  | SegformerB2 | 73.5(3.2↑)    | 52.6(1.8↑) |

​				**Table Ⅳ**   Evaluation metrics on PST900 obtained from compared methods. The best results are shown in bold.

| Model                       | Backbone    | PST900 Dataset |              |
| --------------------------- | ----------- | -------------- | ------------ |
|                             |             | mAcc(%)        | mIoU(%)      |
| MFNet <sub>17</sub>[60]     | -           | 63.50          | 50.34        |
| ACNet <sub>19</sub>[17]     | ResNet-50   | 78.67          | 71.81        |
| RTFNet <sub>19</sub>[62]    | ResNet-152  | 65.69          | 60.46        |
| PSTNet <sub>20</sub>[61]    | ResNet-18   | -              | 68.36        |
| ABMDRNet <sub>21</sub>[63]  | ResNet-50   | 79.06          | 71.33        |
| MMDRNet <sub>23</sub>[64]   | ResNet-50   | 81.30          | 68.70        |
| HAFFSeg <sub>23</sub>[65]   | MobileViT   | 96.10          | 83.80        |
| FDCNet <sub>23</sub>[66]    | ResNet-50   | 85.96          | 77.11        |
| MMSMCNet <sub>23</sub>[67]  | MiT-B3      | 95.20          | 79.80        |
| CLSNet-S* <sub>24</sub>[68] | SegformerB2 | 94.59          | 78.41        |
| MMDNet <sub>24</sub>[69]    | ResNet-50   | 91.04          | 74.62        |
| MDBFNet <sub>24</sub>[70]   | ResNet-50   | 92.50          | 84.90        |
| RSFNet <sub>24</sub>[71]    | ResNet101   | 85.70          | 79.10        |
| MPMTNet-T                   | SegformerB4 | 95.74          | 81.87        |
| MPMTNet-S                   | SegformerB2 | 94.01          | 74.71        |
| MPMTNet-KD                  | SegformerB2 | 95.11(1.1↑)    | 76.19(1.48↑) |

### Acknowledgement

The implement of this project is based on the codebases bellow.💪With thanks to all the workers open code for their help.🙏

- [Segformer](https://github.com/NVlabs/SegFormer)
- [DGPINet-KD]([XUEXIKUAIL/DGPINet](https://github.com/XUEXIKUAIL/DGPINet))



### 
