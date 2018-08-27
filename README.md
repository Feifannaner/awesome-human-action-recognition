# awesome-human-action-recognition
list the most popular methods about human action recognition
## Table of Contents
 - [arXiv Papers](#arxiv-papers)
 - [Journal Papers](#journal-papers)
 - [Conference Papers](#conference-papers)
   - 2018: [CVPR](#2018-cvpr), [ECCV](#2018-eccv), [Others](#2018-others)
   - 2017: [CVPR](#2017-cvpr), [ICCV](#2017-iccv), [Others](#2017-others)
   - 2016: [CVPR](#2016-cvpr), [ECCV](#2016-eccv), [Others](#2016-others)
   - 2015: [CVPR](#2015-cvpr), [ICCV](#2015-iccv), [Others](#2015-others)
   - 2014: [CVPR](#2014-cvpr), [Others & Before](#2014-others--before)
 - [Datasets](#datasets)
 - [Workshops](#workshops)
 - [Challenges](#challenges)
 - [Other Related Papers](#other-related-papers)

## arxiv Papers
##### [\[arXiv:1807.00898\]](https://arxiv.org/abs/1808.07507) Model-based Hand Pose Estimation for Generalized Hand Shape with Appearance Normalization. [\[PDF\]](https://arxiv.org/pdf/1808.07507.pdf  )

## Journal Papers

##### \[2018 IEEE Access\] SHPR-Net: Deep Semantic Hand Pose Regression From Point Clouds. [\[PDF\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8425735&tag=1)
_Xinghao Chen, Guijin Wang, Cairong Zhang, Tae-Kyun Kim, Xiangyang Ji_

## Conference Papers

### 2018 ECCV

##### HandMap: Robust Hand Pose Estimation via Intermediate Dense Guidance Map Supervision. [\[PDF\]](http://yongliangyang.net/docs/handMap_eccv18.pdf)
_Xiaokun Wu, Daniel Finnegan, Eamonn O'Neill, Yongliang Yang_

### 2018 CVPR

##### Learning Pose Specific Representations by Predicting Different Views. [\[PDF\]](https://arxiv.org/pdf/1804.03390.pdf)  [\[Project Page\]](https://poier.github.io/PreView/)  [\[Code\]](https://github.com/poier/PreView)
_Georg Poier, David Schinagl, Horst Bischof_

### 2018 Others

##### [2018 ICPR] Local Regression Based Hourglass Network for Hand Pose Estimation from a Single Depth Image. \[PDF\]
_Jia Li, Zengfu Wang_

### 2017 ICCV
##### Learning to Estimate 3D Hand Pose from Single RGB Images. [\[PDF\]](https://arxiv.org/pdf/1705.01389.pdf)  [\[Project Page\]](https://lmb.informatik.uni-freiburg.de/projects/hand3d/)   [\[Code\]](https://github.com/lmb-freiburg/hand3d)
_Christian Zimmermann, Thomas Brox_


## Datasets

- S/R: Synthetic (S) or Real (R) or Both (B)
- C/D: Color (RGB) or Depth (D)
- Obj: Interaction with objects or not
- #J:  No. of joints
- V: view (3rd or egocentric)
- #S: No. of subjects
- #F: No. of frames (train/test)

|Dataset|Year|S/R|C/D|Obj|#J|V|#S|#F|Related Paper|
|------|------|------|------|------|------|------|------|------|----------------|
|[NYU Hand Pose](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm) | 2014 | R | D | No | 36 | 3rd | 2 | 72k/8k | Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks, SIGGRAPH 2014 [\[PDF\]](http://cims.nyu.edu/~tompson/others/TOG_2014_paper_PREPRINT.pdf)|
|[ICVL](https://labicvl.github.io/hand.html) | 2014 | R | D | No |  16 | 3rd  |  10 | 331k/1.5k | Latent regression forest: Structured estimation of 3d articulated hand posture, CVPR 2014 [\[PDF\]](https://labicvl.github.io/docs/pubs/Danny_CVPR_2014.pdf)|
|[MSRA15](https://github.com/geliuhao/CVPR2016_HandPoseEstimation/issues/4) | 2015 | R | D | No |  21 | 3rd  |  9 | 76,375 | Cascaded Hand Pose Regression, CVPR 2015 [\[PDF\]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf)|
|[BigHand2.2M](http://icvl.ee.ic.ac.uk/hands17/challenge/) | 2017 | R | D | No |  21 | 3rd  |  10 | 2.2M | Big Hand 2.2M Benchmark: Hand Pose Data Set and State of the Art Analysis, CVPR 2017 [\[PDF\]](https://labicvl.github.io/docs/pubs/Shanxin_CVPR_2017.pdf)|
|[EgoDexter](http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm) | 2017 | R | C+D | Yes |  5 | ego  |  4 | 1485 | Real-time Hand Tracking under Occlusion from an Egocentric RGB-D Sensor, ICCV 2017 [\[PDF\]](http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/content/OccludedHands_ICCV2017.pdf)|
|[SynthHands](http://handtracker.mpi-inf.mp
## workshops
## challeges
## other related works
