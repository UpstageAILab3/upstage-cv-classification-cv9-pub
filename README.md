# Document Type Classification | 문서 타입 분류
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
```bash
pip install -r requirements.txt
```

## 1. Competiton Info
### 개요
- 이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다.
- 총 17개의 클래스가 존재.
- train set은 1570개의 이미지로 구성.
- test set은 3140개의 이미지로 구성.

### 평가 지표
F1 score는 Precision과 Recall의 조화 평균을 의미합니다. 클래스마다 개수가 불균형할 때 모델의 성능을 더욱 정확하게 평가할 수 있습니다.

<img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Files/01555d7c-ad8a-4ce3-9692-33d2be0eaaf6.png" width="380">

### Timeline
2024.07.30 ~ 2024.08.11


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
#### 학습데이터
![](https://github.com/SUNGMYEONGGI/image/blob/main/TrainDataset%20Image.png?raw=true)

#### 테스트데이터
![](https://github.com/SUNGMYEONGGI/image/blob/main/TestDataset%20Image.png?raw=true)

#### 이미지 분포
![](https://github.com/SUNGMYEONGGI/image/blob/main/%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%91%E1%85%A9.png?raw=true)
- 데이터 레이블의 분포 시각화
- 전체 이미지 사이즈의 분포를 확인 후 Resize 기준 잡음

![](https://github.com/SUNGMYEONGGI/image/blob/main/%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%8C%E1%85%B3%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%91%E1%85%A9.png?raw=true)
- 사이즈 분포를 히스토그램으로 시각화한 결과
- 대략 350-680 픽셀 높이와 400-750 픽셀 너비 범위에 분포

### Data Processing
#### 학습데이터 전처리
![](https://github.com/SUNGMYEONGGI/image/blob/main/%E1%84%92%E1%85%AE%E1%86%AB%E1%84%85%E1%85%A7%E1%86%AB%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%8C%E1%85%B3%E1%86%BC%E1%84%80%E1%85%A1%E1%86%BC%20%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%8E%E1%85%A5%E1%84%85%E1%85%B5.png?raw=true)
- 90도 각도만 돌리다가 각도가 다양해지면서 결과가 좋아짐, 각도도 세분화 학습함.
    - 90 -> 45 -> 30도 각도
- 블러, 노이즈, 플립을 추가하고 이후 조합된 내용으로 증강
- 이후 밝기와 대비, CLAHE 효과 등을 추가 진행

![](https://github.com/SUNGMYEONGGI/image/blob/main/%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9.png?raw=true)
- 노이즈가 많은 데이터를 확인하고 제거함으로서 학습을 높임
- 1570장에서 1200장으로 줄였고 학습에 조금 더 나은 성능을 만듦
- 불필요하고 노이즈가 낀 데이터가 줄어듦으로 인해 학습량이 줄어 시간 줆

#### 테스트데이터 전처리
![](https://github.com/SUNGMYEONGGI/image/blob/main/%E1%84%83%E1%85%B5%E1%84%82%E1%85%A9%E1%84%8B%E1%85%B5%E1%84%8C%E1%85%B5%E1%86%BC.png?raw=true)
- 가우시안 노이즈를 임의로 추가
- 제거하는 방향으로 학습, SIDD와 같은 Open데이터 셋을 학습한 Pretrained 모델도 사용하여 노이즈 제거를 실시

![](https://github.com/SUNGMYEONGGI/image/blob/main/deskew%20%E1%84%92%E1%85%AA%E1%86%AF%E1%84%8B%E1%85%AD%E1%86%BC.png?raw=true)
![](https://github.com/SUNGMYEONGGI/image/blob/main/OCR%E1%84%92%E1%85%AA%E1%86%AF%E1%84%8B%E1%85%AD%E1%86%BC.png?raw=true)
- deskew 라이브러리 활용
    - deskew 라이브러리로 이미지를 1차적으로 평행하게 맞춤
    - 방향이 8가지로 줄어듦
- OCR 라이브러리 활용
    - 샤프닝 필터와 Non-Local Means Denoising 적용하여 이미지 전처리 후 이미지를 8가지로 돌려가면서 paddleocr 라이브러리를 통해 이미지 OCR 진행
    - 가장 OCR이 많이 된 이미지를 올바른 이미지로 선택
    - 결과적으로 70%가 올바르게 돌아감

![](https://github.com/SUNGMYEONGGI/image/blob/main/%E1%84%8F%E1%85%A2%E1%84%82%E1%85%B5.png?raw=true)
- 이미지 회전을 위해 이미지 내부에서 특이점을 찾아서 반영하여 이미지를 바로 세움
- 빈도가 높은 직선과 사각형을 기준으로 회전

## 4. Modeling 
### Model descrition
#### Efficentnet
> EfficientNetb2부터 EfficientNetb7 테스트 진행했으며 높은 숫자의 모델로 갈수록 높은 해상도에 해당하는 이미지 인식 작업에 유리하며, 이번 대회의 이미지가 고해상도의 임지가 아니여서 실질적으로 높은 모델이 필요하지 않다고 판단하였음

> *모델링의 조건*
> ```img_size = 224
> LR = 1e-4
> num_workers = 1
> epoach = 5
> ```

<img src="/Users/seongmyeong-gi/Desktop/upstage-cv-classification-cv9-pub/img/efficientnet567.png" width="450" height="350">

- Efficientbet_b5
    - `pretrain = True`로 진행하여 초반부터 91%의 f1 score를 기록
    - epoch 3에서 0.9914점으로 early stopping 하면서 모델 테스트 종료
    - 해당 모델 제출 결과 0.9026 

- Efficientbet_b6
    - Efficientnet_b6부터는 `pretrained = False`로 진행
    - epoch 5에서 0.9116 모델 테스트 종료
    - 해당 모델로 제출하지 않고 정답데이터와 비교 했을 때 f1 score 0.75

- Efficientbet_b7
    - `pretrained = False`로 진행
    - epoch 5에 0.8342 모델 테스트 종료
    - 해당 모델로 제출하지 않고 정답데이터와 비교 했을 때 f1 score 0.75


#### VIT
> 여타 CNN 모델과는 다른 Transformer를 활용한 이미지 분류 모델이며, NLP 모델에 자주 쓰인 Transformer를 활용한 모델인 만큼 문서 분류에 활용될 수 있을 것이라 기대를 했으며 VIT의 여러가지 모델 중 3가지를 테스트 진행

> *모델링의 조건*
> ```img_size = 224
> LR = 1e-4
> num_workers = 1
> epoach = 5
> ```

<img src="/Users/seongmyeong-gi/Desktop/upstage-cv-classification-cv9-pub/img/vit.png" width="450" height="350">

- vit_base_patch16_224
    - `pretrain = True`로 진행
    - epoch 5에서 0.9802 모델 테스트 종료
    - 해당 모델로 제출했을 때 f1 score 0.8388

- vit_large_patch16_224
    - `pretrain = True`로 진행
    - epoch 4에 0.9800에 도달하여 모델 테스트 종료
    - 해당 모델로 제출 f1 score 0.8844로 높은 점수를 보이나 1회 epoch에 19분 정도 소요


## 5. Result
### Public Score
![Public Score](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/Public%20Score.png)
### Private Score
![Private Score](https://raw.githubusercontent.com/SUNGMYEONGGI/image/main/Private%20Score.png)

### Presentation
- [[패스트캠퍼스] Upstage AI Lab 3기_CV 경진대회_발표자료_9조](https://drive.google.com/file/d/1rFdmwU4g3G_VGTso5HJ3L0jimqNG32cx/view?usp=sharing)

## 6. etc
### Reference
- https://www.kaggle.com/datasets/pdavpoojan/the-rvlcdip-dataset-test/data
- https://deep-learning-study.tistory.com/212
- https://dream-and-develop.tistory.com/316