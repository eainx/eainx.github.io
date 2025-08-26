---
layout: post
title: OminiControl 리뷰
tags: Transformer, DL, GenerativeModel
published: true
math: true
date: 2025-04-08 09:00 +0900
---


[https://github.com/Yuanshi9815/OminiControl](https://github.com/Yuanshi9815/OminiControl)

> **OminiControl: Minimal and Universal Control for Diffusion Transformer (2024)**
> 

## 🔍 연구 개요

- **Image Control**
    - 텍스트 조건 방식보다 이미지 조건 방식이 정확한 안내가 될 수 있음
- **기존 ‘이미지 조건’ 방식의 문제점 (Image control method)**
    1. **모델 크기 증가**: 많은 파라미터를 추가해야 함
    2. **제한된 범용성**: 특정한 제어 작업만 잘됨 (공간 정렬 / 공간 비정렬)
    3. **UNet 기반 중심**: 최신 DiT 아키텍처와는 잘 안 맞음

## 🍎 Related works

- **Diffusion models**
    - DiT
    - FLUX (SOTA)
- **Controllable generation**
    - ControlNet (공간 정렬 제어)
    - T2I-Adapter (경량 모델)
    - UniControl (moE, 공간 정렬 제어 경량 모델)
    - IP-Adapter (공간 비정렬 제어, cross-attention)
    - SSR-Encoder (공간 비정렬 제어)
    
    → 공간 정렬/비정렬 제어를 둘 다 하는 모델은 없음
    

## ✅ 핵심 기여

- **Minimal Design (0.1% 파라미터만 추가)**
    - DiT의 기존 VAE 인코더와 Transformer 블록을 재활용
    - LoRA (Low-Rank Adaptation)를 이용한 미세 조정만으로 효율적인 제어 가능
- **Unified Sequence Processing**
    - 조건 이미지(예: 엣지, 깊이, 주제 등)를 토큰으로 변환하여 이미지 토큰과 직접 연결 (concatenation)
    - 기존 방식은 단순히 피쳐를 더했지만, 이 방식은 유연한 cross-token attention이 가능
- **Dynamic Position Encoding**
    - 공간 정렬된 제어 (ex: depth, canny) → 동일 위치 인덱스
    - 공간 비정렬 제어 (ex: 주제 기반 생성) → 위치를 일정량 오프셋
    
    → 이를 통해 다양한 제어 작업에 모두 적응 가능
    
- **Condition Strength Control**
    - attention 계산에 bias 행렬 B(γ)를 도입해서 조건 강도 조절이 가능
    - γ 값에 따라 조건의 영향을 키우거나 줄일 수 있어 유연성 증가
- **Subjects200K 데이터셋**
    - 주제 기반 생성을 위한 200K개 이미지 페어로 구성된 새로운 데이터셋
    - GPT-4o + FLUX.1 모델을 활용해 정체성 유지하면서 다양한 장면을 생성.
    - 평가와 필터링도 GPT-4o로 수행해 높은 품질 유지

## ✨ 세부 내용

### 🐻‍❄️ Preliminary

- **DiT**
    - Transformer를 Diffusion의 디노이징 네트워크로 사용해 noisy image token을 refine함
    - FLUX.1, Stable Diffusion 3, PixArt
- token의 종류 (d = embedding, N = 이미지 토큰 개수, M = 텍스트 토큰 개수)
    - noisy image token X ∈ R N×d
    - text condition token Ct ∈ R M×d
- **FLUX.1**
    - 각 DiT 블록은 Layer normalization + Multimodal attention (Rotary Position Embedding, RoPE을 포함)으로 이루어져 있음
    - **RoPE (Rotary position embedding)**
        - 왜 하는가?
            - 기존 RPE (Relative Postion Embedding) 방식은 Relative position 정보를 **더하는** 방식인데, 이러면 내적을 할 때 (qTk) 두 벡터 간의 상대 거리 (각도)가 보존됨
            - 즉, position index로 weight한 일정량으로 회전시켜 임베딩하면, 효과적으로 두 벡터 간의 상대 거리를 나타낼 수 있음
            
        - 이미지 토큰 X에 rotation matrix R(i,j)를 적용함 (token의 2D그리드상  위치 (i,j)에 따라)
        - 텍스트 토큰 Ct에는 position을 (0,0)으로 놓고 같은 rotation 적용 (공간 정보가 없기 때문에)
            
    - **Multimodal attention**
        - positional encoding된 토큰들을 query Q, key K, value K로 project시켜, attention계산
        - [X;Ct]는 이미지와 텍스트의 concat을 말함 (bidirectional attention)x

### ■ Minimal Design

💡 **parameter reuse strategy**

- VAE 인코더 재활용: DiT가 사용하는 기존 VAE 인코더로 조건 이미지를 latent 공간으로 인코딩
    - 이 latent는 DiT가 원래 쓰는 noisy image token과 형태가 같음 → 그대로 사용 가능
    - Transformer 블록은 그대로 유지
- LoRA (Low-Rank Adaptation) 를 활용해 일부 weight만 소량 학습 (약 0.1% 파라미터 증가)

🔧 핵심 아이디어는 기존 자원을 최대한 재사용하면서도 성능을 유지하거나 향상시킴

### ■ **Unified Sequence Processing**

💡 **기존 문제점**

ControlNet이나 T2I-Adapter 같은 모델은 보통 condition feature를 더하는 방식을 사용함

- hX ← hX + hCI
    - hX = noisy image feature, hCI = condition feature
- spatial alignment가 있을 때만 유효하고, 비정렬 조건(ex: subject-driven generation)에는 비효율적
- condition과 image token간의 잠재적 interaction을 제한함

**💡 해결책**
- condition 이미지도 text처럼 **token화해서 (condition token) transformer의 input으로 직접 concat**
    
    ```
    [X; CT; CI]
    [image tokens; text tokens; condition tokens]
    ```
    
- DiT는 **multi-modal attention**을 사용하므로, 모든 토큰 간의 **자유로운 attention 연결**이 가능함
    - 이 덕분에 **spatially-aligned** (e.g. edge-to-image)와 **non-aligned** (e.g. subject-driven) 모두 잘 작동함

📉 실험 결과: 이 방식이 기존 feature add 방식보다 loss가 더 낮고, attention map도 더 의미 있게 나옴

### ■ **Dynamic Position Encoding**

💡 **기존 문제점**


Transformer는 RoPE (Rotary Position Embedding)을 통해 토큰의 위치를 인식하지만, condition token은 spatial align이 아닐 수 있어서 문제 생김

- 각 이미지/텍스트 토큰은 (i, j) 인덱스를 가짐
    - 원본 이미지가 512 x 512일때, VAE 인코더는 32 x 32 latent 토큰을 생성 (i, j ∈ [0, 31])
    - condition 토큰도 똑같이 noisy 이미지 토큰의 인덱스 (i, j)를 가짐
    - 그런데, subject-driven 같은 비정렬 task에서는 **조건 토큰과 이미지 토큰의 위치 인덱스가 겹쳐서 문제가 생김 (위에 펭귄 사라짐)**

💡 **해결책**

두 가지 케이스를 나눔:

- **공간 정렬된 경우 (e.g. Canny → 이미지)**: 조건 토큰과 noisy image 토큰이 **같은 위치 인덱스** 사용
- **비정렬 조건 (e.g. 주제 기반 생성)**: 조건 토큰의 위치를 **일정 오프셋 (e.g. +32)** 줘서 겹치지 않게 함
    - RoPE에서는 그러면 rotation angle이 더 커지는 것..?

📈 결과:

- position을 오프셋 주면 비정렬 task의 loss 감소 + 성능 향상
    
    → 한 모델이 모든 task에 적응 가능한 범용성 확보
    

### ■ **Condition Strength Control**

**💡 기존 문제점**

- 예전 모델은 hX ← hX + α·hCI처럼 **스케일을 조절**해 condition 영향력(세기)을 다뤘음
→ 하지만 OminiControl의 unified attention 방식은 이게 어려움

💡 **해결책**

- attention 연산에서 bias 행렬 B(γ)를 추가해 조건 토큰 간의 attention을 조절:
    
    `bias = torch.log(attn.c_factor[0])`
    ```
    MMA = softmax(QKᵀ / √d + B(γ)) · V
    ```
    
- γ가 클수록 **조건 영향이 커지고**, 0이면 조건 영향 제거 가능

🧪 실험 결과: γ를 조절하면서 원하는 대로 조건의 영향력을 **실시간으로 조절 가능** → 유저가 더 유연하게 제어할 수 있음

### ■ **Subjects200K 데이터셋**

기존 데이터셋은 규모가 작고 다양성이 부족함

📦 Subjects200K 생성 파이프라인

1. **Prompt Generation (주제 생성)**
    - GPT-4o를 이용해서 30,000개 이상의 다양한 subject 설명 생성
    - 각 설명은 하나의 인물이 여러 장면에 등장하는 설정
2. **Paired-image Synthesis (이미지 생성)**
    - 생성된 subject 설명들을 template 구조의 prompt로 재구성 (→ Figure S3 참고)
    - FLUX.1에 입력해서 동일 인물의 서로 다른 장면을 그린 이미지 쌍 생성
3. **Quality Assessment (품질 평가)**
    - 다시 GPT-4o를 활용해 이미지 쌍을 평가
    - 정체성이 어긋나거나 품질이 낮은 페어는 제거해서 고품질 보장

## 🧪 Experiment

### ■ Fine tuning

- LoRA (default rank of 4) 사용
- non condition token을 할때는 0으로 사용

### ■ 학습 세팅 및 환경

| 항목 | 설정 |
| --- | --- |
| 배치 크기 | 실제 batch size = 1 |
| gradient accumulation | 8 step → **effective batch size = 8** |
| **optimizer** | Prodigy Optimizer 사용 |
| 옵션 | warmup, bias correction 활성화 |
| weight decay | 0.01 |
| GPU | 2× NVIDIA H100 (80GB) |
| Spatially-aligned 학습 | 50,000 iteration |
| Subject-driven 학습 | 15,000 iteration |

### **■ Spatially Aligned Tasks**

- FLUX.1-dev
- **Text-to-Image-2M**의 마지막 30만 개 이미지
- 비교 대상 (Baselines)
    
    
    | 모델 | 기반 |
    | --- | --- |
    | ControlNet | Stable Diffusion 1.5 |
    | T2I-Adapter | Stable Diffusion 1.5 |
    | **ControlNetPro** | FLUX.1 기반 ControlNet |
- 평가 지표
    - **Controllabiltiy**
        - F1 / MSE
            
    - **Image Quality**
        - FID (이미지 품질 및 다양성)
        - SSIM (구조 유사도)
        - CLIP-IQA (시각 품질 평가)
        - MAN-IQA
        - MUSICQ
        - PSNR
    - **Alignment (텍스트/이미지-이미지 사이 유사도)**
        - CLIP Text
        - CLIP Image
- 결과 요약
    - OminiControl이 모든 지표에서 ControlNet/T2I-Adapter보다 우수
    - 특히 MSE에서 최대 93% 향상
    - CLIP-IQA도 가장 높게 나옴 → 사람이 보기에도 더 나은 이미지
    - FID에서도 낮은 수치 (더 좋은 품질) 달성

### ■ Subject-Driven Generation

- FLUX.1-schnell (For better visual quality)
- 직접 제작한 **Subjects200K** 데이터셋 사용
- 비교 대상 (Baselines)
    
    
    | 모델 | 기반 |
    | --- | --- |
    | IP-Adapter | FLUX.1 기반 |
- 평가 지표
    - Material quality
    - Identity preservation
    - Color fidelity
    - Natural apperance
    - Modification accuracy
- 결과 요약
    - OminiControl이 기존 IP-Adapter 대비 모든 지표에서 우수
    - Identity preservation: 82.3%
    - Modification accuracy: 90.7% → 조건 반영을 매우 잘함

### ❗한계점

- “파라미터 효율성은 좋지만, **토큰 수가 늘어나서 추론 속도에 부담이 생긴다**”는 점이 한계임
