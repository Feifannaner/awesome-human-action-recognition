# awesome-human-action-recognition
list the most popular methods about human action recognition
## Table of Contents
 - [arXiv Papers](#arxiv-papers)
 - [Journal Papers](#journal-papers)
 - [Conference Papers](#conference-papers)
   - 2019：[ICCV](#2019-iccv),
   - 2018: [CVPR](#2018-cvpr), [ECCV](#2018-eccv), [NIPS](#2018-nips)，[Others](#2018-others)
   - 2017: [CVPR](#2017-cvpr), [ICCV](#2017-iccv), [Others](#2017-others)
   - 2016: [CVPR](#2016-cvpr), [ECCV](#2016-eccv), [Others](#2016-others)
   - 2015: [CVPR](#2015-cvpr), [ICCV](#2015-iccv), [Others](#2015-others)
   - 2014: [CVPR](#2014-cvpr), [Others & Before](#2014-others--before)
 - [Directions](#Directions):
   - [Traditional Machine Learning Methods](#Traditional-machine-learning-methods)
   - [Deep Learning Methods](#traditional-machine-learning-methods)
     - [2D Convolutional Netwoks](#2d-convolutional-netwoks)
     - [3D Convolutional Networks](#3D-convolutional-neetwoks)
     - [LSTM Networks](#LSTM-networks)
     - [Multi-Stream Networks](#multistream-networks)
     - [New feature](#new-feature)
     - [Explanation of deep representation](#explanation-deep-representation)
     - [Semantic](#semantic)
     - [New Datasets](#datasets)


 - [Datasets](#datasets)
 - [Current Accuracy on Main Datasets](#current-accuracy-on-main-datasets)
 - [Workshops](#workshops)
 - [Challenges](#challenges)
 - [Other Related Papers](#other-related-papers)

## arxiv Papers
##### [\[arXiv:1808.07507\]](https://arxiv.org/abs/1808.07507) Model-based Hand Pose Estimation for Generalized Hand Shape with Appearance Normalization. [\[PDF\]](https://arxiv.org/pdf/1808.07507.pdf  )
Unaiza Ahsan,Rishi Madhok
##### [\[arXiv:1711.04161\]](https://arxiv.org/abs/1711.04161) End-to-end Video-level Representation Learning for Action Recognition. [\[PDF\]](https://arxiv.org/pdf/1711.04161.pdf )[\[code\]](https://github.com/zhujiagang/DTPP) 
Jiagang Zhu, Wei Zou, Zheng Zhu
## Journal Papers
##### [\[2017 IEEE Access:TPAMI\]](https://ieeexplore.ieee.org/document/7940083/) Long-Term Temporal Convolutions for Action Recognition [\[PDF\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7940083)
Gul Varol  , Ivan Laptev, and Cordelia Schmid, Fellow, IEEE
## Review works
##### Human Action Recognition and Prediction: A Survey [\[PDF\]](http://cn.arxiv.org/pdf/1806.11230v2)
Yu Kong, Member, IEEE, and Yun Fu, Senior Member, IEEE

## Conference Papers
### 2019 ICCV
Graph Convolutional Networks for Temporal Action Localization
作者：Chuang Gan 等

Action recognition with spatial-temporal discriminative filter banks
作者：Yuanjun Xiong 等

AssembleNet: Searching for Multi-Stream Neural Connectivity in Video Architectures
作者：Google Brain

neural architecture search for video understanding——大力出奇迹

DynamoNet: Dynamic Action and Motion Network
作者：Ali Diba Luc Van Gool

Reasoning About Human-Object Interactions Through Dual Attention Networks
作者：Bolei Zhou

Learning Temporal Action Proposals with Fewer Labels
作者：Stanford Feifei组 Juan Carlos Niebles

EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition
作者：Dima Damen 等

SlowFast Networks for Video Recognition
（文章链接：https://arxiv.org/abs/1812.03982）
kaiming 大神 from FAIR 

Video Classification with Channel-Separated Convolutional Networks
（文章链接：https://arxiv.org/abs/1904.02811）
Du Tran 大神 from FAIR

SCSampler: Sampling Salient Clips from Video for Efficient Action Recognition. oral
（文章链接：https://arxiv.org/abs/1904.04289）
Du Tran 大神 from FAIR

DistInit: Learning Video Representations without a Single Labeled Video.
（文章链接：https://arxiv.org/abs/1901.09244）
Du Tran 大神 from FAIR 
很简单的思路

TSM: Temporal Shift Module for Efficient Video Understanding
作者：Ji Lin, Chuang Gan, Song Han
论文链接：https://arxiv.org/abs/1811.08383
Github链接：https://github.com/mit-han-lab/temporal-shift-module
emmm感觉吧，就像是搞了个带Mask的固定卷积核？

BMN: Boundary-Matching Network for Temporal Action Proposal Generation
（文章链接：https://arxiv.org/abs/1907.09702）
来自作者大大解读：林天威：[ICCV 2019][时序动作提名] 边界匹配网络详解
（原文链接：https://zhuanlan.zhihu.com/p/75444151）

Weakly Supervised Energy-Based Learning for Action Segmentation.oral
文章链接：https://github.com/JunLi-Galios/CDFL

Pose-aware Dynamic Attention for Human Object Interaction Detection
文章链接：https://github.com/bobwan1995/PMFNet

What Would You Expect? Anticipating Egocentric Actions With Rolling-Unrolling LSTMs and Modality Attention
项目链接：https://iplab.dmi.unict.it/rulstm/
论文链接：https://arxiv.org/pdf/1905.09035.pdf
GitHub：https://github.com/fpv-iplab/rulstm

Fine-Grained Action Retrieval Through Multiple Parts-of-Speech Embeddings
论文链接：https://arxiv.org/abs/1908.03477
项目链接：https://mwray.github.io/FGAR/

HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips
作者：Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, Josef Sivic
论文链接：https://arxiv.org/abs/1906.03327
项目链接：https://github.com/antoine77340/howto100m
code（链接：https://github.com/antoine77340/howto100m）

Temporal Attentive Alignment for Large-Scale Video Domain Adaptation
作者：Min-Hung Chen, Zsolt Kira, Ghassan AlRegib, Jaekwon Woo, Ruxin Chen, Jian Zheng
论文链接：https://arxiv.org/abs/1907.12743
Github链接：https://github.com/cmhungsteve/TA3N

STM- SpatioTemporal and Motion Encoding for Action Recognition
from ZJU && SenseTime Group Limited 
论文链接：https://arxiv.org/abs/1908.02486
### 2018 ECCV
##### [2018,ECCV] Temporal Relational Reasoning in Videos [\[PDF\]](http://people.csail.mit.edu/bzhou/publication/eccv18-TRN.pdf) [\[code\]](https://github.com/metalbubble/TRN-pytorch)
##### [2018,ECCV] Modality Distillation with Multiple Stream Networks for Action Recognition [\[PDF\]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nuno_Garcia_Modality_Distillation_with_ECCV_2018_paper.pdf) 
Bolei Zhou, Alex Andonian, Aude Oliva, and Antonio Torralba
##### [2018,ECCV] Graph Distillation for Action Detection with Privileged Modalities [\[PDF\]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zelun_Luo_Graph_Distillation_for_ECCV_2018_paper.pdf)
Stanford University 2 Google Inc.
##### above two papers, they are similar, which belong to a new hole 
##### [2018,ECCV] Spatio-Temporal Channel Correlation Networks for Action Classification [\[PDF\]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ali_Diba_Spatio-Temporal_Channel_Correlation_ECCV_2018_paper.pdf)
##### note: qustion:3D network cannot learn the relation between spacial and temporal .why?
##### [2018,ECCV] Learning Human-Object Interactions by Graph Parsing Neural Networks [\[PDF\]](https://arxiv.org/pdf/1808.07962.pdf) [\[code\]](https://github.com/SiyuanQi/gpnn)
Siyuan Qi, Wenguan Wang, Baoxiong Jia, Jianbing Shen, Song-Chun Zhu
##### [2018,ECCV] Interaction-aware Spatio-temporal Pyramid Attention Networks for Action Classification[\[PDF\]](https://arxiv.org/pdf/1808.01106.pdf)
Yang Du,Chunfeng Yuan, Bing Li, Lili Zhao, Yangxi Li and Weiming Hu
##### [2018,ECCV] Action Search: Spotting Actions in Videos and Its Application to Temporal Action Localization[\[PDF\]](https://arxiv.org/pdf/1706.04269.pdf)
Humam Alwassel, Fabian Caba Heilbron, and Bernard Ghanem
##### [2018,ECCV] Action Anticipation with RBF Kernelized Feature Mapping RNN [\[PDF\]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yuge_Shi_Action_Anticipation_with_ECCV_2018_paper.pdf)
Yuge Shi, Basura Fernando, Richard Hartley
##### [2018,ECCV] Skeleton-Based Action Recognition with Spatial Reasoning and Temporal Stack Learning[\[PDF\]](https://arxiv.org/pdf/1805.02335.pdf)
Chenyang Si, Ya Jing, Wei Wang, Liang Wang, Tieniu Tan
##### [2018,ECCV] Scenes-Objects-Actions: A Multi-Task, Multi-Label Video Dataset 
Jamie Ray, Heng Wang, Du Tran, Yufei Wang, Matt Feiszli, Lorenzo Torresani, Manohar Paluri
##### [2018,ECCV] End-to-End Joint Semantic Segmentation of Actors and Actions in Video [\[PDF\]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jingwei_Ji_End-to-End_Joint_Semantic_ECCV_2018_paper.pdf)
##### [2018,ECCV] Scenes-Objects-Actions: A Multi-Task, Multi-Label Video Dataset [\[PDF\]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Heng_Wang_Scenes-Objects-Actions_A_Multi-Task_ECCV_2018_paper.pdf)
Jamie Ray1, Heng Wang1, Du Tran1 Yufei Wang1 ,etc
### 2018 CVPR
##### [2018,CVPR] Optical Flow Guided Feature: A Fast and Robust Motion Representation for
Video Action Recognition [\[PDF\]](https://arxiv.org/pdf/1711.11152.pdf)
Shuyang Sun, Zhanghui Kuang, Wanli Ouyang, Lu Sheng, Wei Zhang
##### [2018,CVPR] Appearance-and-Relation Networks for Video Classification [\[PDF\]](https://arxiv.org/pdf/1711.09125.pdf) [\[code\]](https://github.com/wanglimin/ARTNet)
L. Wang, W. Li, W. Li, and L. Van Gool 
### 2018 NIPS
##### [2018,NIPS] Trajectory Convolution for Action Recognition[\[PDF\]](https://papers.nips.cc/paper/7489-trajectory-convolution-for-action-recognition) [\[code\]](https://github.com/metalbubble/TRN-pytorch)
Yue Zhao， Yuanjun，Xiong
### 2018 Others

### 2017 ICCV


### 2017 CVPR
##### AdaScan: Adaptive Scan Pooling in Deep Convolutional Neural Networks for Human Action Recognition in Videos [\[PDF\]](https://arxiv.org/pdf/1611.08240.pdf)
Amlan Kar, Nishant Rai， Karan Sikka,Gaurav Sharma
##### [2017,CVPR] On the Integration of Optical Flow and Action Recognition [\[PDF\]](https://arxiv.org/pdf/1712.08416.pdf)
Laura Sevilla-Lara, Yiyi Liao, Fatma Guney, Varun Jampani, Andreas Geiger, Michael J. Black
### 2017 Others

### 2016 CVPR
##### [2016,CVPR] Convolutional Two-Stream Network Fusion for Video Action Recognition[\[PDF\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780582)
Christoph Feichtenhofer，Axel Pinz，Andrew Zisserman


##### [2016,CVPR] A Key Volume Mining Deep Framework for Action Recognition[\[PDF\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780588)
Wangjiang Zhu,Jie Hu,Gang Sun,Xudong Cao,Yu Qiao

### 2016 ECCV
##### [2016,ECCV] Temporal Segment Networks: Towards Good Practices for Deep Action Recognition [\[PDF\]](http://cn.arxiv.org/pdf/1608.00859)
Limin Wang,Yuanjun XiongZhe WangYu QiaoDahua LinXiaoou TangLuc Van Gool
### 2016 ICCV
### 2016 Others

### 2015 CVPR
##### [2015,CVPR] Action Recognition with Trajectory-Pooled Deep-Convolutional Descriptors [\[PDF\]](http://cn.arxiv.org/pdf/1505.04868v1)
Limin Wang, Yu Qiao, Xiaoou Tang
### 2015 ECCV
### 2015 ICCV
##### [2015,ICCV] Learning Spatiotemporal Features with 3D Convolutional Networks [\[PDF\]](https://arxiv.org/pdf/1412.0767.pdf)
D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri
### 2015 Others

### 2014 CVPR
##### [2014,CVPR] Large-Scale Video Classification with Convolutional Neural Networks [\[PDF\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6909619)
A Karpathy ， G Toderici ， S Shetty ， T Leung ， R Sukthankar,L. Fei-Fei
### 2014 ECCV
### 2014 ICCV
### 2014 Others

##### [2014,NIPS] Two-Stream Convolutional Networks for Action Recognition in Videos[\[PDF\]](http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf)
Karen Simonyan,  Andrew Zisserman
#### 
Two-Stream Convolutional Networks for Action Recognition in Videos
##### AdaScan: Adaptive Scan Pooling in Deep Convolutional Neural Networks for Human Action Recognition in Videos [\[PDF\]](http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf)
Karen Simonyan,  Andrew Zisserman
## Directions
### Traditional Machine Learning Methods
Here we pay more attention on DL methods as follows.
### Deep Learning Methods
#### 2D convolutional netwoks
##### AdaScan: Adaptive Scan Pooling in Deep Convolutional Neural Networks for Human Action Recognition in Videos [\[PDF\]](https://arxiv.org/pdf/1611.08240.pdf)
Amlan Kar, Nishant Rai， Karan Sikka,Gaurav Sharma

#### 3D convolutional networks
##### [\[2014,IEEE Acess:TPAMI\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6165309) 3D Convolutional Neural Networks for Human Action Recognition
Shuiwang Ji ,Wei Xu,Ming Yang ,Kai Yu
##### [\[2017 IEEE Access:TPAMI\]](https://ieeexplore.ieee.org/document/7940083/) Long-Term Temporal Convolutions for Action Recognition [\[PDF\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7940083)
Gul Varol  , Ivan Laptev, and Cordelia Schmid, Fellow, IEEE
#### LSTM networks

#### multistream networks
##### [2014,NIPS] Two-Stream Convolutional Networks for Action Recognition in Videos[\[PDF\]](http://papers.nips.cc/paper/5353-two-stream-convolutional)
##### [2016,ECCV] Temporal Segment Networks: Towards Good Practices for Deep Action Recognition [\[PDF\]](http://cn.arxiv.org/pdf/1608.00859)
##### [2017,ICCV] Temporal Relational Reasoning in Videos [\[PDF\]](http://people.csail.mit.edu/bzhou/publication/eccv18-TRN.pdf) [\[code\]](https://github.com/metalbubble/TRN-pytorch)
##### [2016,CVPR] A Key Volume Mining Deep Framework for Action Recognition[\[PDF\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780588)
#### new feature
##### [2018,CVPR] Optical Flow Guided Feature: A Fast and Robust Motion Representation for
Video Action Recognition [\[PDF\]](https://arxiv.org/pdf/1711.11152.pdf)
_Shuyang Sun, Zhanghui Kuang, Wanli Ouyang, Lu Sheng, Wei Zhang
##### [2015,CVPR] Action Recognition with Trajectory-Pooled Deep-Convolutional Descriptors [\[PDF\]](http://cn.arxiv.org/pdf/1505.04868v1)
##### [2017,CVPR] On the Integration of Optical Flow and Action Recognition [\[PDF\]](https://arxiv.org/pdf/1712.08416.pdf)
Laura Sevilla-Lara, Yiyi Liao, Fatma Guney, Varun Jampani, Andreas Geiger, Michael J. Black

####  explanation deep representation
##### [\[arXiv:1712.08416\]](https://arxiv.org/pdf/1712.08416.pdf) What have we learned from deep representations for action recognition?
Laura Sevilla-Lara, Yiyi Liao, Fatma Guney, Varun Jampani, Andreas Geiger, Michael J. Black
#### semantic 
[\[arXiv:1802\]](https://arxiv.org/pdf/1802.06459.pdf) Structured Label Inference for Visual Understanding
Nelson Nauata, Hexiang Hu, Guang-Tong Zhou, Zhiwei Deng, Zicheng Liao and Greg Mori
#### datasets
 ##### [2018,ECCV] Scenes-Objects-Actions: A Multi-Task, Multi-Label Video Dataset [\[PDF\]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Heng_Wang_Scenes-Objects-Actions_A_Multi-Task_ECCV_2018_paper.pdf)

## Datasets
- Year： publish date
- Videos: amount of flips
- Views: amount of view angles
- Actions: amount of action class
- Subjects: people in Videos
- Modility: RGB or RGB-D
- Env: Controlled(C) or Uncontrolled(U)
##### dataset papers 2017 [\[PDF\]](https://www.researchgate.net/publication/236156020_A_survey_of_video_datasets_for_human_action_and_activity_recognition)
##### 2018 video benchmarks: a review[\[PDF\]](https://www.researchgate.net/publication/327078109_Video_benchmarks_of_human_action_datasets_a_review)
##### video datasets online(html)[\[HTML\]](https://www.di.ens.fr/~miech/datasetviz/)
##### compute vision datasets online[\[HTML\]](https://projet.liris.cnrs.fr/voir/wiki/doku.php?id=datasets)

|Dataset|Year|Videos|Views|Actions|Subjects|Modility|Env(C\U)|Related Paper|
|------|------|------|------|------|------|------|------|----------------|
|[KTH](http://www.nada.kth.se/cvap/actions/) | 2004 | 599 | 1 | 6 | 25 | RGB | C | Recognizing human actions: A local svm approach, IEEE ICPR 2004 [\[PDF\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1334462)|
|[HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) | 2011| 7000 | - | 51 |  - | RGB  |  U | LHmdb: A large video database for human motion recognition, ICCV 2011 [\[PDF\]](http://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/Kuehne_etal_iccv11.pdf)|
|[UCF101](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) | 2012| 13320 | - | 101 |  - | RGB  |  U | Ucf101: A dataset of 101 human action classes from videos in the wild, 2012,cRCV-TR-12-01 [\[PDF\]](https://arxiv.org/pdf/1212.0402.pdf)|
## Current Accuracy on Main Datasets
- [HDMB51](http://actionrecognition.net/files/dsetdetail.php?did=5;) 82.1% 2017 
## workshops
## challeges
## other related works
