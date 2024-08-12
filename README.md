[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FVjNDCrt)
# Title (Please modify the title)
## Team

| ![임동건](https://avatars.githubusercontent.com/u/125024589?v=4) | ![김주형](https://avatars.githubusercontent.com/u/95218618?v=4) | ![성명기](https://avatars.githubusercontent.com/u/104310191?v=4) | ![유정수](https://avatars.githubusercontent.com/u/50096716?v=4) | ![장재성](https://avatars.githubusercontent.com/u/165862584?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            팀장, 데이터증강, 모델링          |          오토인코더, Test데이터 디노이징      |           데이터 증강, 모델링     |    데이터 OCR         |                모델링 테스트               |
|            [임동건](https://github.com/LimDG1981)             |            [김주형](https://github.com/AjouKim)             |            [성명기](https://github.com/SUNGMYEONGGI)             |            [유정수](https://github.com/Dream-Forge-Studios)             |            [장재성](https://github.com/mirrbandi)             |

## 0. Overview
### Environment
-   AMD Ryzen Threadripper 3960X 24-Core Processor
-   NVIDIA GeForce RTX 3090
-   CUDA Version 12.2

### Requirements    
-   timm==0.9.12
-   torch==2.1.0
-   torchvision=0.16.0
-   numpy==1.26.0
-   scikit-learn=1.3.2

## 1. Competiton Info
### Overview
 이번 대회는 computer vision domain에서 가장 중요한 태스크인 이미지 분류 대회입니다.

 이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다. 이러한 이미지 분류는 의료, 패션, 보안 등 여러 현업에서 기초적으로 활용되는 태스크입니다. 딥러닝과 컴퓨터 비전 기술의 발전으로 인한 뛰어난 성능을 통해 현업에서 많은 가치를 창출하고 있습니다.

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/f35917ed-effd-4c5d-8f79-10fe1718bcc7)
  
  그 중, 이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용될 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/e69229b9-b3c1-443b-a5c2-2ce499667c89)

이번 대회를 통해서 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다. computer vision에서 중요한 backbone 모델들을 실제 활용해보고, 좋은 성능을 가지는 모델을 개발할 수 있습니다. 그 밖에 학습했던 여러 테크닉들을 적용해 볼 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.
-   **input** : 3140개의 이미지
-   **output** : 주어진 이미지의 클래스

### 평가 지표
F1 score는 Precision과 Recall의 조화 평균을 의미합니다. 클래스마다 개수가 불균형할 때 모델의 성능을 더욱 정확하게 평가할 수 있습니다. 수식은 다음과 같습니다
 
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/253cd5a2-0806-4822-8135-e5b35b8a88e3)
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/4b52b801-89df-4e6c-b86c-48219fde4c1e)
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/6dd9eedb-2c05-46cd-a6fd-80cf19d40b42)

- [참고자료](https://www.v7labs.com/blog/f1-score-guide)
Macro F1 score는 multi classification을 위한 평가 지표로 클래스 별로 계산된 F1 score를 단순 평균한 지표입니다.
    
    ![image](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/01555d7c-ad8a-4ce3-9692-33d2be0eaaf6.png)

## 2. Components

### Directory
```
├── JANG JAESEONG
│   └── CODE
├── KIM JUHYUNG
│   └── CODE
├── LIM DONGGUN
│   └── CODE
├── SEONG MYEONGGI
│   └── CODE
├── YU JEONGSU
│   └── CODE
└── README.md
```

## 3. Data descrption

### Dataset overview
![Dataset](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/Dataset%20%E1%84%80%E1%85%A2%E1%84%8B%E1%85%AD.png)

### EDA
실제 데이터의 형상을 확인함
![image](https://github.com/user-attachments/assets/22ec492a-995b-4023-837d-26bfe8ddd6ad)

데이터 레이블의 분포를 시각화 하였고, 전체 이미지 사이즈의 분포를 확인하여 Resize 기준을 잡음
![EDA](https://github.com/user-attachments/assets/d55daaf7-a786-416d-ba88-cce9cd30e3bb)

사이즈 분포를 히스토그램으로 시각화한 결과
![image](https://github.com/user-attachments/assets/16e00ca4-6016-440e-884a-77e4c08d678c)


### Data Processing

#### TRAIN 데이터 전처리
![image](https://github.com/user-attachments/assets/384e5edf-a952-4077-825b-3cbc6d4c82a1)
다양한 이미지 데이터를 사전에 변형해봄으로써 테스트 이미지와 같이 만들어 보며 학습에 필요한 내용을 확인함.

![image](https://github.com/user-attachments/assets/3f599871-6ae8-47a7-8796-9b41c97f1c07)
데이터 증강, 처음에는 90도 각도만 돌리다가 각도가 다양해지면서 결과가 좋아져서 각도도 세분화 학습함.
90 -> 45 -> 30도 각도
블러, 노이즈, 플립을 추가하고 이후 조합된 내용으로 증강을 진행함.
이후에도 밝기와 대비, CLAHE 효과 등을 추가하여 진행하였음.

![image](https://github.com/user-attachments/assets/d853af4b-bae5-4013-ab77-7975c254c4d8)

주어진 TRAIN 데이터에도 증가하기전에 노이즈가 많은 데이터를 확인하고 제거함으로서 학습을 높였음.
처음 주어진 데이터가 1570장이었는데 1200장으로 줄였고 학습에 조금 더 나은 성능을 만들어주었음.
불필요하고 노이즈가 낀 데이터가 줄어듦으로 인해서 학습량이 줄어 시간도 단축되었음.
데이터 가공에 있어서 모델학습을 위한 데이터 처리의 중요함을 생각하게 되었음.

#### TEST 데이터 전처리

다른 시도로는 테스트 데이터의 노이즈를 처리하였음.
![EDA2](https://github.com/user-attachments/assets/ed7f2caf-9e24-4d4d-bd1f-8bcd8ee22989)
원본에 가우시안 노이즈를 임의로 추가한후, 제거하는 방향으로 학습하였고
SIDD와 같은 Open데이터 셋을 학습한 Pretrained 모델도 사용하여 노이즈 제거를 실시하였음.

학습된 모델이 더 잘 작동하기 위해서 테스트 데이터 역시 전처리가 필요하고 이에 따라 분류 성능이 좌우됨.
![image](https://github.com/user-attachments/assets/435f0656-4bbf-4500-97b6-484caef2ddc9)

![image](https://github.com/user-attachments/assets/9a17f396-3f45-49b8-84a9-c97115d7de25)
이미지 회전을 위해서 이미지 내부에서 특이점을 찾아서 반영하여 이미지를 바로 세우는 시도를 함.

![image](https://github.com/user-attachments/assets/8c73a0e9-b297-4e0a-b2af-2e05b79097fe)
여러 시도 끝에 데이터 전체를 회전하였음. 
데이터 회전에는 성공하였지만 시간상 다시 이것을 가지고 모델을 학습하고 진행하는 것을 체계화 하지 못한 것은 아쉬움.

## 4. Modeling 
### Model descrition
![efficentnet](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/cv-project-modeltest-img1.png)
![vit_model](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/cv-project-modeltest-img3.png)

### Modeling Process
![image](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/cv-project-modeltest-img2.png)
![image](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/cv-project-modeltest-img4.png)
![image](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/cv-project-modeltest-img5.png)

## 5. Result

### Leader Board
#### Public Score
![Public Score](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/Public%20Score.png)
#### Private Score
![Private Score](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/Private%20Score.png)

### Presentation

- [[패스트캠퍼스] Upstage AI Lab 3기_CV 경진대회_발표자료_9조](https://drive.google.com/file/d/1rFdmwU4g3G_VGTso5HJ3L0jimqNG32cx/view?usp=sharing)

## 6. etc
### Reference

- https://www.kaggle.com/datasets/pdavpoojan/the-rvlcdip-dataset-test/data
- https://deep-learning-study.tistory.com/212
- https://dream-and-develop.tistory.com/316

