# Meta-RL

# 🌟 DynaMITE-RL & HLT-Dynamite-RL

이 레포지토리는 **DynaMITE-RL(NeurIPS 2024)** 알고리즘의 GridWorld 구현과,
이를 확장해 새로운 latent 구조를 실험하는 **HLT-DynaMITE-RL 버전**을 포함하고 있다.
핵심 원리는 모두 논문 내용을 기반으로 한다. 

---

# 📁 1. Directory Structure

```
meta_rl/
│
├── dynamite_rl/                # 원 논문 구조를 따르는 DynaMITE-RL 구현
│   ├── checkpoints/            # 저장된 모델 가중치
│   ├── logs/                   # 학습 과정 로그
│   ├── agent.py                # PPO 에이전트 + latent belief update
│   ├── config.py               # 실험 설정 (learning rate, latent dim 등)
│   ├── envs.py                 # GridWorld 기반 DLCMDP 환경
│   ├── main.py                 # DynaMITE-RL 학습 전체 파이프라인
│   ├── models.py               # VAE encoder/decoder, policy/value network
│   └── train_YYYYMMDD_xxxxxx.log
│
├── hlt_dynamite_rl/            # HLT-DynaMITE-RL: latent 구조 확장 실험
│   ├── checkpoints_hlt/        # HLT 실험용 모델 저장
│   ├── graph/                  # 학습곡선, best policy evaluation 그래프
│   ├── logs/                   # HLT 버전 로그
│   ├── agent.py                # HLT-DynaMITE-RL agent (top/mid latent 지원)
│   ├── config.py               # HLT 실험 설정
│   ├── envs.py                 # 동일 GridWorld 환경 (HLT 용)
│   ├── main.py                 # HLT-DynaMITE-RL 학습 파이프라인
│   └── models.py               # top/mid latent 구조 모델 정의
│
└── README.md
```

---

# 🔍 2. What is Implemented?

## ✔ DynaMITE-RL (논문 원형 재현)

* Session 단위로 변하는 latent context (DLCMDP)
* Variational inference 기반 latent posterior 업데이트
* Consistency loss 적용
* Previous posterior → next prior (latent belief conditioning)
* Session masking으로 reconstruction 안정화

## ✔ HLT-DynaMITE-RL

* **v1:** top latent only
* **v2:** mid-latent가 top latent 학습을 보조
* latent disentanglement 및 RL 성능 비교 실험

---

# 🧠 3. Key Files Overview

### `envs.py`

* 5×5 GridWorld
* 두 goal 중 하나만 session마다 +1
* DLCMDP termination(p_switch) 적용

### `models.py`

* GRU 기반 encoder
* Gaussian posterior ( q(z|\tau) )
* session termination head
* state/reward decoder
* PPO actor/critic

### `agent.py`

* rollout 중 posterior 업데이트
* consistency + masked ELBO 계산
* PPO advantage 계산

### `main.py`

* 학습 전체 파이프라인
* rollout → VAE update → PPO update
* checkpoint & logging

---

# 🚀 4. How to Run

### DynaMITE-RL

```
cd meta_rl/dynamite_rl
python main.py
```

### HLT-DynaMITE-RL

```
cd meta_rl/hlt_dynamite_rl
python main.py
```

---

# 📈 5. Logs & Graphs

* `logs/` : 학습 로그
* `checkpoints*/` : 모델 가중치
* `graph/` : HLT-DynaMITE-RL 학습곡선 및 best-policy reward 곡선

  * mean ± std shading 포함

그래프 제목 예시:
**Best Policy Evaluation Reward Over Training**

---

# ⚙️ 6. Configuration

각 config.py에서 설정 가능:

* latent dimension
* learning rate
* PPO hyperparameters
* p_switch
* training steps
* seed

---

# 📚 7. Reference

본 구현은 다음 논문에 기반한다:
**DynaMITE-RL: A Dynamic Model for Improved Temporal Meta-Reinforcement Learning (NeurIPS 2024)**
전체 알고리즘·수식·ELBO 도식은 PDF 참조. 


