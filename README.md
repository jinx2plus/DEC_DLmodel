# DEC_DLmodel

이 저장소는 교통 검지기 시계열 데이터에 대해 **Deep Embedded Clustering(DEC)** 기반 실험을 정리한 프로젝트입니다.

## 프로젝트 개요

본 프로젝트는 KOTI 교통혼잡 예측 및 신호제어 과제의 실험 결과를 정리한 형태이며, Autoencoder로 특징을 추출한 뒤 DEC(Deep Embedded Clustering)로 교통 패턴을 군집화하는 파이프라인을 제공합니다.

- 입력 데이터: 정규화된 CSV(`input_normalize.csv`)
- 특징 추출: Autoencoder latent space(차원 축소)
- 군집화: DEC(Soft Assignment + KL loss)
- 산출물: 군집 라벨 예측, 정확도/ARI/NMI, 혼동행렬, 결과 엑셀 파일

## 저장소 구성

- `dec_dense_layer.py`
  - 노트북 코드를 실행 가능한 Python 스크립트로 정리한 메인 코드
- `0922Keras_DEC_DenseLayer-project.ipynb`
  - 핵심 실험 노트북
- `Keras_DEC_DenseLayer.ipynb`
  - 보조 실험 노트북
- `090202Keras_DEC_DenseLayer-project-Copy1.ipynb`, `090202Keras_DEC_DenseLayer-project-Copy2.ipynb`
  - 과거 실험본(참고용)
- `metrics.py`
  - `dec_dense_layer.py`에서 사용되는 ACC / NMI / ARI 계산 모듈
- `requirements.txt`
  - 실행 의존성
- `README.md`
  - 프로젝트 설명서
- `.gitignore`
- 결과 산출물 예시:
  - `...pdf`, `...xlsx`, `results/*.h5`

## 설치

```bash
cd F:\DEC_DLmodel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> 권장 환경: Python 3.7+

> 참고: `dec_dense_layer.py`는 `metrics` 모듈을 import합니다. 실행 경로에 `metrics.py`가 있어야 합니다.

## 실행 방법

1) 경로 수정

`dec_dense_layer.py`에서 데이터 입력 경로와 결과 저장 경로를 현재 환경에 맞게 수정합니다.

```python
trains = np.loadtxt("<input_normalize.csv 경로>", skiprows=1, delimiter=',', dtype=float)
...
df.to_excel('<결과 저장 경로>/result.xlsx', index=False)
```

2) 실행

```bash
python dec_dense_layer.py
```

3) 생성 결과 확인

- 가중치: `results/ae_weights.h5`, `results/DEC_model_final.h5`
- 결과 파일: `result.xlsx`
- 학습/평가 지표: Acc, NMI, ARI, Confusion Matrix

## 주요 하이퍼파라미터

- `n_clusters = 5`
  - 기본값(실험 기준). 데이터 성격에 따라 조정 가능
- Autoencoder 구조 예시: `[input_dim, 500, 1000, 2000, 3000, 5000, 10]`
- Optimizer: `SGD`
  - pretrain: `lr=0.1, momentum=0.9`
  - DEC fine-tune: `lr=0.01, momentum=0.9`
- `batch_size = 128`
- `pretrain_epochs = 1000`
- `maxiter = 20000`, `update_interval = 140`, `tol = 0.001`

## 코드 흐름

`dec_dense_layer.py`는 다음 순서로 동작합니다.

1. CSV 로딩 및 전처리
   - 현재 코드 기준: `[:,531:795]` 구간을 사용해 264차원 입력 생성
2. Autoencoder 학습
3. encoder 임베딩에 대해 KMeans로 초기 클러스터 중심 초기화
4. DEC 모델 구성 (`ClusteringLayer`, KL loss)
5. 반복 학습
   - `q`(soft assignment) 계산 후 보조 분포 `p` 갱신
   - `delta_label < tol` 기준으로 조기 종료
6. 테스트 추론 및 지표 계산
7. 결과 엑셀 저장 및 시각화

## 주의사항

- 연구 실험 목적 코드이므로, 운영 적용 전에는 경로, 라벨 정의, 군집 개수, 피처 범위를 재확인하세요.
- 한글 경로/인코딩 이슈가 있으면 영문 경로에서 실행하는 것을 권장합니다.

## 라이선스

별도의 라이선스 파일이 없어 배포 전 라이선스 정책을 지정해 주세요.

---

## English Version

# DEC_DLmodel

This repository contains Deep Embedded Clustering (DEC)-based experiments on traffic detector time-series data.

## Project Overview

This project organizes experimental work from a KOTI traffic congestion prediction and signal control task and provides a pipeline that applies DEC on top of Autoencoder-based feature representations.

- Input: normalized CSV (`input_normalize.csv`)
- Feature extraction: Autoencoder latent space (dimensionality reduction)
- Clustering: DEC (Soft Assignment + KL loss)
- Outputs: cluster predictions, Accuracy/NMI/ARI, confusion matrix, and result spreadsheets.

## Repository Structure

- `dec_dense_layer.py`
  - Main executable Python script converted from the notebook workflow
- `0922Keras_DEC_DenseLayer-project.ipynb`
  - Core experiment notebook
- `Keras_DEC_DenseLayer.ipynb`
  - Supporting experiment notebook
- `090202Keras_DEC_DenseLayer-project-Copy1.ipynb`, `090202Keras_DEC_DenseLayer-project-Copy2.ipynb`
  - Legacy experiment copies for reference
- `metrics.py`
  - Metrics helper module used by `dec_dense_layer.py` (ACC/NMI/ARI)
- `requirements.txt`
  - Runtime dependencies
- `README.md`
  - Project documentation
- `.gitignore`
- Example outputs:
  - `...pdf`, `...xlsx`, `results/*.h5`

## Setup

```bash
cd F:\DEC_DLmodel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Recommended: Python 3.7+

> Note: `dec_dense_layer.py` imports `metrics`; ensure `metrics.py` is available in the runtime path.

## How to Run

1) Update paths

Modify dataset input and output paths in `dec_dense_layer.py` for your environment.

```python
trains = np.loadtxt("<path to input_normalize.csv>", skiprows=1, delimiter=',', dtype=float)
...
df.to_excel('<path to output>/result.xlsx', index=False)
```

2) Run

```bash
python dec_dense_layer.py
```

3) Check outputs

- Weights: `results/ae_weights.h5`, `results/DEC_model_final.h5`
- Result file: `result.xlsx`
- Logs / metrics: Accuracy, NMI, ARI, Confusion Matrix

## Key Hyperparameters

- `n_clusters = 5`
  - Default from current experiment; adjust based on data and task goals
- Autoencoder architecture example: `[input_dim, 500, 1000, 2000, 3000, 5000, 10]`
- Optimizer: `SGD`
  - pretrain: `lr=0.1, momentum=0.9`
  - DEC finetune: `lr=0.01, momentum=0.9`
- `batch_size = 128`
- `pretrain_epochs = 1000`
- `maxiter = 20000`, `update_interval = 140`, `tol = 0.001`

## Code Pipeline

`dec_dense_layer.py` follows this workflow:

1. Load and preprocess CSV
   - Current code uses the slice `[:,531:795]` to build a 264-dimensional feature vector
2. Train Autoencoder
3. Initialize DEC cluster centers using KMeans on encoder embeddings
4. Build DEC model (`ClusteringLayer`, KL loss)
5. Iterative training
   - Update soft assignments `q` and auxiliary target distribution `p`
   - Stop when convergence condition `delta_label < tol` is met
6. Run inference on test set and compute metrics
7. Export output spreadsheet and visualizations

## Notes

- This is a research-oriented codebase. Before production use, validate input paths, label definitions, number of clusters, and feature ranges.
- If you face path/encoding issues, especially with Korean paths, use ASCII-only directories for stable execution.

## License

No separate license file is provided. Please define your license policy before external distribution.
