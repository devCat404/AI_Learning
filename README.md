# 인공지능 학습 로드맵

## 0. 전체적인 학습 구조

인공지능의 학습 문제를 크게 다음과 같이 분류한다.

1. 지도학습 (Supervised Learning)
2. 비지도학습 (Unsupervised Learning)
3. 준지도학습 (Semi-Supervised Learning)
4. 자기지도학습 (Self-Supervised Learning)
5. 강화학습 (Reinforcement Learning)

본 학습에서는 우선 **지도학습과 비지도학습**을 중심으로 공부한다.

---

# 1. 지도학습 (Supervised Learning)

입력 $X$와 정답 $Y$가 주어진 상태에서

$$
X \rightarrow Y
$$

라는 관계를 학습한다.

일반적으로 다음과 같은 조건을 고려한다.

- 입력 데이터의 형태
- 출력 변수의 형태
- 데이터의 분포
- 데이터의 크기
- 변수의 차원
- 선형성 / 비선형성
- 이상치 (Outlier)
- 결측치 (Missing Value)
- 다중공선성 (Multicollinearity)
- 이분산성 (Heteroscedasticity)
- 클래스 불균형 (Class Imbalance)
- 노이즈 (Noise)
- 과적합 (Overfitting)
- 데이터의 시간적 의존성
- 데이터의 공간적 의존성

---

# 1.1 회귀 (Regression)

연속적인 값을 예측하는 문제.

$$
X \rightarrow Y,\qquad Y \in \mathbb{R}
$$

대표적인 목표:

- 가격 예측
- 온도 예측
- 수요 예측
- 매출 예측
- 확률 예측
- 시간 예측

## 1.1.1 기본 선형 회귀

### Linear Regression

$$
y = \beta_0 + \beta_1x_1 + \cdots + \beta_px_p + \epsilon
$$

대표적인 학습 방법:

- Ordinary Least Squares (OLS)
- Maximum Likelihood Estimation (MLE)

대표적인 손실함수:

$$
L(\beta) = \sum_{i=1}^{N} (y_i-\hat{y}_i)^2
$$

### Ridge Regression

L2 정규화를 사용하는 선형 회귀.

$$
L(\beta) = \sum_{i=1}^{N} (y_i-\hat{y}_i)^2 + \lambda\sum_{j=1}^{p}\beta_j^2
$$

특히 다음과 같은 상황에서 고려한다.

- 다중공선성
- 많은 feature
- 계수의 크기를 안정화해야 하는 경우
- 과적합

### Lasso Regression

L1 정규화를 사용한다.

$$
L(\beta) = \sum_{i=1}^{N} (y_i-\hat{y}_i)^2 + \lambda\sum_{j=1}^{p}|\beta_j|
$$

특징:

- feature selection
- sparse solution
- 많은 변수 중 일부만 중요한 경우

### Elastic Net

L1과 L2 정규화를 결합한다.

$$
L(\beta) = \sum_{i=1}^{N} (y_i-\hat{y}_i)^2 + \lambda_1\sum_j|\beta_j| + \lambda_2\sum_j\beta_j^2
$$

특히:

- feature가 많음
- 변수 사이의 상관성이 높음
- sparse model이 필요함

등의 상황에서 고려한다.

---

# 1.2 Robust Regression

이상치에 민감한 일반적인 Least Squares Regression의 단점을 보완한다.

## 주요 문제

일반적인 MSE는 residual이 커질수록

$$
r^2
$$

에 비례하여 손실이 증가한다.

따라서 일부 매우 큰 이상치가 전체 모델을 강하게 끌어당길 수 있다.

---

## 1.2.1 Huber Regression

작은 residual에서는 제곱 오차를 사용하고,

큰 residual에서는 절대값에 가까운 손실을 사용한다.

$$
L_\delta(r) = \begin{cases} \frac{1}{2}r^2, & |r|\leq\delta \\ 
\delta\left(|r|-\frac{1}{2}\delta\right), & |r|>\delta \end{cases}
$$

적합한 상황:

- 이상치가 존재
- 대부분의 데이터는 정상적
- 이상치를 완전히 제거하고 싶지는 않음

---

## 1.2.2 LAD Regression

Least Absolute Deviations.

$$
L(\beta) = \sum_i |y_i-\hat{y}_i|
$$

특징:

- MSE보다 이상치에 강함
- Median과 밀접한 관계
- Laplace noise 가정과 연결

---

## 1.2.3 RANSAC Regression

정상적인 데이터와 이상치가 섞여 있을 때

- 일부 데이터만 선택
- 모델 추정
- inlier 판단
- 반복

을 수행한다.

적합한 상황:

- 이상치 비율이 높음
- 명확한 inlier 구조가 존재
- 기하학적 데이터
- 센서 데이터
- 컴퓨터 비전

---

# 1.3 Quantile Regression

평균(mean)이 아니라 특정 분위수(quantile)를 예측한다.

$$
Q_Y(\tau|X=x)
$$

여기서

$$
0 < \tau < 1
$$

이다.

예:

- $\tau=0.5$: Median
- $\tau=0.9$: 90th percentile
- $\tau=0.95$: 95th percentile

---

## 1.3.1 Quantile Loss

Pinball Loss:

$$
L_\tau(y,\hat{y}) = \begin{cases} \tau(y-\hat{y}), & y\geq\hat{y} \\ 
(1-\tau)(\hat{y}-y), & y<\hat{y} \end{cases}
$$

적합한 상황:

- 예측 구간이 필요
- 데이터 분포의 비대칭성이 존재
- 평균보다 상위/하위 분위수가 중요
- Risk estimation
- 수요 예측
- 금융
- 보험

---

# 1.4 비선형 회귀 (Nonlinear Regression)

입력과 출력 사이의 관계가 선형이 아닌 경우.

$$
y=f(X)+\epsilon
$$

대표적인 모델:

- Polynomial Regression
- Regression Spline
- Generalized Additive Model (GAM)
- Kernel Regression
- Gaussian Process Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- XGBoost
- LightGBM
- CatBoost
- Support Vector Regression (SVR)
- Neural Network Regression

---

# 1.5 Tree 기반 회귀

## Decision Tree Regression

입력 공간을 여러 영역으로 나누고 각 영역에서 예측값을 결정한다.

적합한 상황:

- 비선형 관계
- feature interaction
- feature scaling이 중요하지 않은 경우
- 해석 가능한 모델이 필요한 경우

---

## Random Forest Regression

여러 Decision Tree를 ensemble한다.

대표적인 장점:

- 비선형 관계 처리
- feature interaction 처리
- 비교적 안정적인 성능
- 과적합 완화
- scaling 불필요

---

## Gradient Boosting Regression

약한 모델을 순차적으로 학습하면서 이전 모델의 오류를 보완한다.

대표적인 모델:

- Gradient Boosting Machine (GBM)
- XGBoost
- LightGBM
- CatBoost

특히 tabular data에서 강력한 baseline이 될 수 있다.

---

# 1.6 Kernel 기반 회귀

## Support Vector Regression (SVR)

margin과 kernel을 이용하여 비선형 회귀를 수행한다.

대표적인 kernel:

- Linear Kernel
- Polynomial Kernel
- RBF Kernel
- Sigmoid Kernel

적합한 상황:

- 데이터 크기가 비교적 작음
- feature dimension이 높음
- 비선형 관계
- kernel similarity가 유용한 경우

---

# 1.7 Gaussian Process Regression

함수 자체를 확률과정으로 모델링한다.

$$
f(x)\sim\mathcal{GP}(m(x),k(x,x'))
$$

특징:

- 예측 평균
- 예측 분산
- uncertainty estimation

을 동시에 얻을 수 있다.

적합한 상황:

- 데이터가 많지 않음
- uncertainty가 중요함
- Bayesian 접근이 필요함
- 실험 설계
- Bayesian Optimization

---

# 1.8 Neural Network Regression

복잡한 비선형 함수를 학습한다.

$$
\hat{y}=f(X;\theta)
$$

대표적인 모델:

- MLP
- Deep Neural Network
- CNN
- RNN
- LSTM
- GRU
- Transformer

입력 데이터의 구조에 따라 적합한 architecture가 달라진다.

---

# 2. 분류 (Classification)

입력 $X$를 하나 이상의 class로 분류한다.

$$
X\rightarrow Y
$$

여기서

$$
Y\in\{1,2,\dots,K\}
$$

이다.

---

# 2.1 Binary Classification

두 개의 class를 구분한다.

예:

- 정상 / 이상
- 스팸 / 정상
- 양성 / 음성

대표 모델:

- Logistic Regression
- Linear SVM
- Kernel SVM
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Neural Network

---

# 2.2 Multiclass Classification

세 개 이상의 class 중 하나를 선택한다.

대표 모델:

- Multinomial Logistic Regression
- Softmax Regression
- Decision Tree
- Random Forest
- SVM
- Gradient Boosting
- Neural Network

---

# 2.3 Multilabel Classification

하나의 입력에 여러 label이 동시에 존재할 수 있다.

예:

$$
x\rightarrow\{cat,dog,outdoor\}
$$

대표적인 접근:

- Binary Relevance
- Classifier Chains
- Label Powerset
- Neural Network + Sigmoid
- Transformer-based models

---

# 2.4 Class Imbalance

class별 데이터 수가 크게 다른 경우.

예:

$$
N_{\text{class A}}\gg N_{\text{class B}}
$$

문제:

- Accuracy가 misleading할 수 있음
- Minority class를 제대로 학습하지 못함
- decision boundary가 majority class에 편향될 수 있음

주요 해결법:

### Data-level

- Oversampling
- Undersampling
- SMOTE
- ADASYN

### Algorithm-level

- Class Weight
- Weighted Loss
- Focal Loss

### Ensemble-level

- Balanced Random Forest
- EasyEnsemble

---

# 3. 지도학습에서 데이터 특성에 따른 모델 선택

## 데이터가 선형적인 경우

고려할 모델:

- Linear Regression
- Logistic Regression
- Ridge
- Lasso
- Elastic Net
- Linear SVM

---

## 데이터가 비선형적인 경우

고려할 모델:

- Polynomial Regression
- Kernel SVM
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Neural Network

---

## Tabular Data

강력한 후보:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Neural Network

특히 tree boosting 계열을 우선적으로 고려한다.

---

## 고차원 데이터

고려할 모델:

- Ridge
- Lasso
- Elastic Net
- Linear SVM
- Kernel SVM
- Neural Network

추가적으로:

- PCA
- Feature Selection
- Regularization

등을 고려한다.

---

## 이상치가 많은 데이터

고려할 모델:

- Huber Regression
- LAD Regression
- RANSAC
- Quantile Regression
- Tree-based models

---

## 데이터의 분산이 일정하지 않은 경우

### Heteroscedasticity

$$
Var(Y|X=x)=\sigma^2(x)
$$

즉,

$$
\sigma^2(x)
$$

가 입력 $x$에 따라 달라지는 경우.

고려할 방법:

- Weighted Least Squares
- Generalized Least Squares
- Quantile Regression
- Heteroscedastic Regression
- Probabilistic Neural Network
- Mixture Density Network

---

# 4. 비지도학습 (Unsupervised Learning)

정답 $Y$가 주어지지 않은 상태에서 데이터의 구조를 찾는다.

$$
X\rightarrow\text{structure}
$$

주요 문제:

1. Clustering
2. Dimensionality Reduction
3. Density Estimation
4. Anomaly Detection
5. Representation Learning
6. Association Rule Learning

---

# 4.1 Clustering

데이터를 비슷한 그룹으로 묶는다.

---

## K-Means

$$
\min_{\{C_k\}}
\sum_{k=1}^{K}
\sum_{x_i\in C_k}
\|x_i-\mu_k\|^2
$$

적합한 상황:

- cluster가 비교적 구형
- cluster 개수를 알고 있음
- 대규모 데이터

---

## Gaussian Mixture Model

데이터를 여러 Gaussian distribution의 mixture로 표현한다.

$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k,\Sigma_k)
$$

특징:

- soft clustering
- 각 cluster에 속할 확률을 계산
- 타원형 cluster 표현 가능

---

## DBSCAN

밀도 기반 clustering.

장점:

- cluster 개수를 미리 지정하지 않아도 됨
- noise/outlier 탐지 가능
- 비구형 cluster 처리 가능

---

## Hierarchical Clustering

cluster를 계층적으로 구성한다.

대표 방법:

- Agglomerative Clustering
- Divisive Clustering

---

# 4.2 Dimensionality Reduction

고차원 데이터를 저차원 공간으로 표현한다.

$$
X\in\mathbb{R}^{D}
\rightarrow
Z\in\mathbb{R}^{d},
\qquad d<D
$$

대표 모델:

- PCA
- Kernel PCA
- ICA
- NMF
- t-SNE
- UMAP
- Autoencoder

---

# 4.3 Density Estimation

데이터가 어떤 확률분포를 가지고 있는지 추정한다.

$$
p(x)
$$

대표 방법:

- Gaussian Distribution
- Gaussian Mixture Model
- Kernel Density Estimation
- Normalizing Flow
- Variational Autoencoder

---

# 4.4 Anomaly Detection

정상적인 데이터에서 벗어난 관측치를 탐지한다.

대표 모델:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Autoencoder
- Variational Autoencoder
- Robust Statistical Methods

---

# 5. 데이터 형태에 따른 모델 선택

## 5.1 Tabular Data

데이터 형태:

$$
X\in\mathbb{R}^{N\times D}
$$

대표 모델:

- Linear Models
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- MLP

---

# 5.2 Time Series

시간 순서가 존재하는 데이터.

$$
x_1,x_2,\dots,x_T
$$

고려할 모델:

### Statistical

- AR
- MA
- ARMA
- ARIMA
- SARIMA
- VAR
- State Space Model

### Machine Learning

- Random Forest
- XGBoost
- LightGBM

### Deep Learning

- RNN
- LSTM
- GRU
- Temporal CNN
- Temporal Transformer

---

# 5.3 Image

공간적 구조가 존재하는 데이터.

대표 모델:

- CNN
- ResNet
- DenseNet
- EfficientNet
- Vision Transformer
- Swin Transformer

주요 문제:

- Image Classification
- Object Detection
- Semantic Segmentation
- Instance Segmentation
- Image Generation

---

# 5.4 Audio

시간에 따른 신호 구조가 존재한다.

대표 표현:

- Waveform
- Spectrogram
- Mel-Spectrogram
- MFCC

대표 모델:

- CNN
- 1D CNN
- 2D CNN
- RNN
- LSTM
- GRU
- CRNN
- Transformer
- Audio Spectrogram Transformer

---

# 5.5 Text

순서와 의미 구조가 중요한 데이터.

대표 모델:

- Bag-of-Words
- TF-IDF
- Word2Vec
- GloVe
- RNN
- LSTM
- GRU
- Transformer
- BERT
- GPT 계열

---

# 6. 데이터 분포에 따른 모델 선택

## Gaussian-like Distribution

고려:

- Linear Regression
- Gaussian Process
- Gaussian Mixture Model

---

## Heavy-tailed Distribution

고려:

- Robust Regression
- Student-t Regression
- Quantile Regression
- Robust Statistical Models

---

## Skewed Distribution

고려:

- Transformation
- Generalized Linear Models
- Quantile Regression
- Tree-based Models

---

## Multimodal Distribution

고려:

- Gaussian Mixture Model
- Mixture Models
- Mixture Density Network
- Clustering

---

## Non-stationary Distribution

특히 시간에 따라 데이터의 통계적 특성이 변하는 경우.

고려:

- Time Series Models
- State Space Models
- Online Learning
- Adaptive Models
- Transformer-based Models

---

# 7. 회귀 문제에서 발생할 수 있는 주요 문제

## 7.1 Outlier

해결:

- Huber Regression
- LAD
- RANSAC
- Quantile Regression

---

## 7.2 Multicollinearity

해결:

- Ridge
- Lasso
- Elastic Net
- PCA

---

## 7.3 Heteroscedasticity

해결:

- Weighted Least Squares
- GLS
- Quantile Regression
- Probabilistic Regression

---

## 7.4 Nonlinearity

해결:

- Polynomial Regression
- GAM
- Kernel Methods
- Tree Models
- Neural Networks

---

## 7.5 Overfitting

해결:

- Regularization
- Cross Validation
- Early Stopping
- Dropout
- Bagging
- Boosting
- Data Augmentation

---

# 8. 분류 문제에서 발생할 수 있는 주요 문제

## Class Imbalance

해결:

- Class Weight
- Oversampling
- Undersampling
- SMOTE
- Focal Loss
- Balanced Ensemble

---

## Overlapping Classes

해결:

- Kernel SVM
- Tree Ensemble
- Neural Network
- Feature Engineering

---

## High-dimensional Data

해결:

- PCA
- Feature Selection
- Regularization
- Linear SVM
- Neural Network

---

## Probability Calibration

분류 모델의 출력 확률이 실제 확률과 일치하지 않을 수 있다.

대표 방법:

- Platt Scaling
- Isotonic Regression
- Temperature Scaling

---

# 9. 모델을 공부할 때의 공통 학습 순서

각 모델은 단순히 "무엇인지"만 공부하지 않고 다음 순서로 학습한다.

## Step 1. 문제 정의

어떤 문제를 해결하는가?

$$
X\rightarrow Y
$$

---

## Step 2. 모델의 가정

모델이 데이터에 대해 어떤 가정을 하는가?

예:

- 선형성
- 독립성
- Gaussian noise
- 동일한 분산
- feature independence

---

## Step 3. 모델의 수학적 형태

예:

$$
y=X\beta+\epsilon
$$

---

## Step 4. Loss Function / Likelihood

예:

$$
L(\theta) = \sum_i (y_i-f(x_i;\theta))^2
$$

또는

$$
\hat{\theta} = \arg\max_\theta \sum_i \log P(y_i|x_i,\theta)
$$

---

## Step 5. 최적화

어떻게 parameter를 학습하는가?

예:

- Closed-form solution
- Gradient Descent
- SGD
- Newton Method
- Coordinate Descent
- EM Algorithm
- SMO

---

## Step 6. 모델의 Bias / Variance

모델이:

- underfitting
- overfitting

중 어디에 취약한지 확인한다.

---

## Step 7. 데이터 특성과의 관계

어떤 데이터에서 강한가?

예:

$$
\text{Outlier}
\rightarrow
\text{Robust Regression}
$$

$$
\text{Multicollinearity}
\rightarrow
\text{Ridge}
$$

$$
\text{Sparse Features}
\rightarrow
\text{Lasso}
$$

$$
\text{Nonlinear Tabular Data}
\rightarrow
\text{Boosting}
$$

$$
\text{Uncertainty}
\rightarrow
\text{Gaussian Process}
$$

---

## Step 8. 실제 모델 비교

각 모델에 대해 다음을 비교한다.

| 항목 | 내용 |
|---|---|
| Problem | 어떤 문제를 해결하는가 |
| Assumption | 어떤 가정을 하는가 |
| Model | 수학적으로 어떻게 표현되는가 |
| Loss | 무엇을 최소화/최대화하는가 |
| Optimization | 어떻게 학습하는가 |
| Strength | 어떤 데이터에서 강한가 |
| Weakness | 어떤 상황에서 약한가 |
| Complexity | 계산량은 어느 정도인가 |
| Interpretability | 해석하기 쉬운가 |
| Robustness | 이상치 등에 얼마나 강한가 |
| Uncertainty | 예측 불확실성을 제공하는가 |

---

# 10. 최종적으로 구축할 학습 트리

```text
Artificial Intelligence
│
├── Supervised Learning
│   │
│   ├── Regression
│   │   ├── Linear Regression
│   │   ├── Ridge
│   │   ├── Lasso
│   │   ├── Elastic Net
│   │   ├── Robust Regression
│   │   │   ├── Huber
│   │   │   ├── LAD
│   │   │   └── RANSAC
│   │   ├── Quantile Regression
│   │   ├── GAM
│   │   ├── SVR
│   │   ├── Decision Tree
│   │   ├── Random Forest
│   │   ├── Gradient Boosting
│   │   ├── XGBoost
│   │   ├── LightGBM
│   │   ├── CatBoost
│   │   ├── Gaussian Process
│   │   └── Neural Network
│   │
│   └── Classification
│       ├── Logistic Regression
│       ├── LDA
│       ├── QDA
│       ├── Naive Bayes
│       ├── SVM
│       ├── Decision Tree
│       ├── Random Forest
│       ├── AdaBoost
│       ├── Gradient Boosting
│       ├── XGBoost
│       ├── LightGBM
│       ├── CatBoost
│       └── Neural Network
│
├── Unsupervised Learning
│   │
│   ├── Clustering
│   │   ├── K-Means
│   │   ├── GMM
│   │   ├── DBSCAN
│   │   └── Hierarchical Clustering
│   │
│   ├── Dimensionality Reduction
│   │   ├── PCA
│   │   ├── Kernel PCA
│   │   ├── ICA
│   │   ├── t-SNE
│   │   ├── UMAP
│   │   └── Autoencoder
│   │
│   ├── Density Estimation
│   │   ├── KDE
│   │   ├── GMM
│   │   ├── VAE
│   │   └── Normalizing Flow
│   │
│   └── Anomaly Detection
│       ├── Isolation Forest
│       ├── One-Class SVM
│       ├── LOF
│       └── Autoencoder
│
├── Semi-Supervised Learning
│
├── Self-Supervised Learning
│
└── Reinforcement Learning
```

# 11. 핵심 학습 방향

각 모델을 독립적으로 암기하기보다는 다음 관계를 이해하는 것을 목표로 한다.

$$
\boxed{
\text{Problem}
\rightarrow
\text{Data Characteristics}
\rightarrow
\text{Assumptions}
\rightarrow
\text{Model}
\rightarrow
\text{Loss/Likelihood}
\rightarrow
\text{Optimization}
\rightarrow
\text{Strength/Weakness}
}
$$

궁극적으로는 다음과 같은 질문에 답할 수 있도록 학습한다.

> "이 데이터에서 어떤 문제가 발생하고 있으며,  
> 그 문제를 해결하기 위해 어떤 모델을 선택해야 하고,  
> 왜 그 모델이 적합한가?"

이 관점에서 각 모델을 공부하면 단순한 알고리즘 암기가 아니라 **모델 선택 능력(Model Selection)**을 중심으로 머신러닝 전체를 연결할 수 있다.
