# DECmodel

이 저장소는 `0922Keras_DEC_DenseLayer-project.ipynb` 노트북을 정리해 GitHub 업로드에 맞게 구성한 프로젝트입니다.

## 프로젝트 개요
- 데이터 과학/딥러닝 실험 노트북을 보존하고, 실행 가능한 형태로 코드 파일을 분리해 제공합니다.
- 노트북(`.ipynb`)과 스크립트(`.py`)를 함께 제공해 재현 가능성을 높입니다.
- Python 기반 의존성은 `requirements.txt`로 관리합니다.

## 파일 구성
- `0922Keras_DEC_DenseLayer-project.ipynb` : 원본 노트북
- `dec_dense_layer.py` : 노트북의 코드 셀만 추출한 실행 스크립트
- `requirements.txt` : 실행에 필요한 패키지 목록
- `.gitignore` : Git에서 제외할 파일 목록

## 사용 방법
1. 가상환경 생성 및 활성화
2. `pip install -r requirements.txt`
3. 노트북 사용: `jupyter notebook` 또는 `jupyter lab`
4. 스크립트 실행: `python dec_dense_layer.py`

## Repository purpose
This repository contains `0922Keras_DEC_DenseLayer-project.ipynb` organized for GitHub upload and reproducible sharing.

## Project overview
- It preserves the original notebook and provides a separated executable code file for reuse.
- The notebook (`.ipynb`) and script (`.py`) are both included to improve reproducibility.
- Python dependencies are tracked in `requirements.txt`.

## Repository structure
- `0922Keras_DEC_DenseLayer-project.ipynb`: original notebook
- `dec_dense_layer.py`: extracted code cells from the notebook
- `requirements.txt`: required package list
- `.gitignore`: ignored files for Git

## How to run
1. Create and activate a virtual environment
2. `pip install -r requirements.txt`
3. Use notebook: `jupyter notebook` or `jupyter lab`
4. Or run script: `python dec_dense_layer.py`

@TODO: dataset and output paths may need local path adjustment before execution.

🚦 Traffic Pattern Clustering with DEC
> Deep Learning based Traffic Signal Optimization Project
> 딥러닝 기반 도심지 교통 패턴 군집화 및 신호 최적화 솔루션

---

## ⚡️ Project at a Glance
본 프로젝트는 한국교통연구원(KOTI) 의 "딥러닝 기반 도심지 교통혼잡 예측 및 신호제어 솔루션 시스템 개발" 과제의 일환으로 수행되었습니다.

기존의 정적인 신호 제어(Static TOD) 방식의 한계를 극복하기 위해, Deep Embedded Clustering (DEC) 알고리즘을 활용하여 시시각각 변하는 교통 패턴을 자율적으로 학습하고 분류하는 딥러닝 모델을 구축했습니다.

---

## 🧐 Why This Project? (Problem Solving)

### 🚫 The Problem
* 정적 운영: 기존 교통 신호는 사전에 정해진 시간표(TOD)대로만 운영되어, 돌발적인 혼잡이나 날씨 변화에 유연하게 대처하지 못함.
* 전문가 의존: 신호 운영 시간대(첨두/비첨두 등)를 구분할 때 데이터보다는 전문가의 경험적 판단에 의존함.

### ✅ The Solution
* Data-Driven: 44개 검지기에서 수집된 대규모 시공간 데이터를 활용.
* Auto Clustering: DEC 알고리즘을 통해 교통량, 속도, 밀도 패턴을 스스로 학습하여 "현재 교통 상황이 어떤 상태인지" 정확히 분류.
* Dynamic Control: 분류된 패턴에 맞춰 최적의 신호 제어 시나리오(Dynamic TOD)를 매칭할 수 있는 기반 마련.

---

## 🧠 Methodology : DEC (Deep Embedded Clustering)

고차원의 시계열 교통 데이터를 효과적으로 군집화하기 위해 Autoencoder와 K-Means가 결합된 DEC 모델을 설계했습니다.

### 1️⃣ Stacked Autoencoder (Dimensionality Reduction)
* Role: 노이즈가 많은 264차원의 원본 데이터를 10차원의 잠재 공간(Latent Space) 으로 압축하여 핵심 특징(Feature)만 추출.
* Structure: Input(264) → Dense(500-1000-2000-3000-5000) → Latent(10)

### 2️⃣ Soft Assignment & Fine-tuning
* Initialization: 압축된 데이터(Z-space)에 K-means를 적용하여 초기 중심점 설정.
* Optimization: Student's t-distribution을 기반으로 데이터와 군집 중심 간의 유사도를 계산하고, KL Divergence를 최소화하며 군집 성능을 강화.

---

## 📊 Dataset & Preprocessing

| Feature | Description |
| :--- | :--- |
| Source | 대전광역시 대덕대로 (대덕대교~경성큰마을 네거리) 44개 검지기 |
| Metrics | 🚗 Volume (교통량), 🚀 Speed (속도), 📦 Density (밀도) |
| Scale | 10분 단위 집계 (06:00 ~ 21:50) × 35일간의 데이터 |
| Preprocessing | Missing Value Handling, Min-Max Normalization |

---

## 🏆 Key Results

### 🎯 Traffic Pattern Discovery
모델 학습 결과, 데이터가 단순 시간대가 아닌 실제 도로 혼잡 강도에 따라 명확하게 군집화되었습니다.

* Cluster A (Morning Peak): 출근 시간대 특유의 고밀도·저속 패턴 식별.
* Cluster B (Off-Peak): 낮 시간대 원활한 흐름 식별.
* Cluster C (Event Driven): 특정 요일/이벤트 발생 시의 비정상 혼잡 패턴 감지.

> 💡 Impact: 이 결과는 VISSIM 시뮬레이션과 연동되어, 각 군집(패턴)별 최적 신호 주기를 산출하는 "AI 기반 신호 최적화 라이브러리" 구축의 핵심 엔진으로 활용되었습니다.

---

## 🛠 Tech Stack

### Languages & Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)

### Data Analysis & Visualization
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=flat-square&logo=matplotlib&logoColor=black) ![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![KOTI](https://img.shields.io/badge/KOTI-Research-0056D2?style=for-the-badge)](https://www.koti.re.kr/)


---
<div align="center">
  <sub>Built with 💻 by YongJin Park for KOTI Research Project</sub>
</div>