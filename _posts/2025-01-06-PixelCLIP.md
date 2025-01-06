---
layout: post
title: PixelCLIP 리뷰
tags: CLIP, SemanticSegmentation
published: true
math: true
date: 2025-01-06 09:00 +0900
---

# PixelCLIP: Towards Open-Vocabulary Semantic Segmentation Without Semantic Labels


[PixelCLIP](https://cvlab-kaist.github.io/PixelCLIP/)

### 1. Introduction

- **기존 기법의 한계 (Semantic Label)**
    - CLIP과 같은 vision-language model을 활용
    - 이미지에서 물체가 ‘무엇’인지는 알 수 있지만 ‘어디’있는지는 알 수 없음
    - CLIP이 이미 물체를 감지하는 데에 효과적이므로 implicit하게 물체의 위치를 학습
- **PixelCLIP의 장점 (Mask + Prompt Learning)**
    - **마스크**
        - semantic label을 사용하는 대신 CLIP이 주목해야 하는 위치를 알려줌
        - DINO와 SAM 같은 vison foundation model을 사용하여 이미지를 나누고, ‘어디’를 볼 지 알려줌 (마스크 생성)
        - 마스크를 통해 CLIP의 **이미지 인코더를** 파인튜닝함
    - **프롬프트 러닝**
        - CLIP의 **텍스트 인코더에서** prompt learning을 통해 learnable class를 구축함
        - learnable class를 centroid로 하여 online clustering을 수행함
        - learnable class는 global하게 모든 이미지에서 적용됨

### 2. Related Work

![{437467F8-FFDD-4853-BB5C-A7537412E0CB}.png](https://github.com/user-attachments/assets/4d1ae3d6-b7de-48d8-8d24-990e589ed70c)

- Open-vocabulary **semantic segmentation**
    - 픽셀 당 semantic label을 활용하므로 computational cost가 비쌈
    - 이를 해결하기 위해, **weakly-supervised setup**을 사용하여 densely-annotate 할 필요 없이 image-level label 혹은 label 없이 사용
    - semantic label보다 vision foundation model을 통해 마스크를 얻고, 이를 CLIP 이미지 인코더를 파인튜닝하기 위한 요소로 사용함
- Fine-tuning vision language models for **dense prediction**
    - CLIP은 image-level supervision (with caption)으로 학습되었으므로 dense prediction에 약함, global image에 주목함 (** 이미지에서 물체가 ‘무엇’인지는 알 수 있지만 ‘어디’있는지는 알 수 없음)
    - OWL-ViT & CAT-Seg: CLIP 인코더를 파인튜닝
    - ZegCLIP & Xu et al: 모델 전체를 파인튜닝 하는 대신, CLIP의 인코더를 위한 global prefix로 수행되는 프롬프트 러닝을 수행함
    - 좋은 성능을 보이나, densely annotate된 GT가 학습에 필요함
- **Vision foundation models**
    - DINO/SAM: fine grained image segmentation
    - semantic label 없이 생성되어, 그냥 마스크만 생성

### 3. Methodology

![{1B78E0D6-A73B-4851-9BBA-D26BF9455879}.png](https://github.com/user-attachments/assets/dbe9ea1d-0e26-46da-94b3-5f752938cebc)

1. **Preliminaries**
    - **이미지 I** ∈ R^(HxWx3), 픽셀 당 semantic label
    - semantic label은 **semantic class T_i=1~S에** S개의 textual description을 줌
    - **CLIP 텍스트 인코더** E_L 는 semantic class T를  받아 텍스트 feature f_T=E_L(T) ∈ R^(S x d)를 얻음
        - S = semantic class의 개수, d = hidden dimension
    - **CLIP 이미지 인코더**는 Dense 이미지 feature  f_I = E_V(I) ∈ R^(H x W x d)
        - H x W = output frame resolution
    - Image-text similarity map M_IT  ∈ R^(H x W x S) (코사인 similarity)
        - 이를 위해 normalize 수행 (||vector||=1)
    - 이것은 이미지와 텍스트 feature 에서 추정된 soft binary mask 라고도 볼 수 있으며, binary mask loss L_mask가 쓰임
    
    **→ 이미지의 픽셀 당 semantic label이 주어지고, 이것은 이미지당 총 S개의 semantic class가 됨**
    
    **→ CLIP 텍스트 인코더는 이것을 S x d의 텍스트 feature로 변환**
    
    **→ CLIP 이미지 인코더는 H x W x d의 이미지 feature을 도출**
    
    **→ 이미지-텍스트 코사인 유사도 맵 (H x W x S), 이것은 soft binary mask임**
    
    ```python
    cost_volume = torch.einsum(
        "nc, bhwc -> bnhw",
        F.normalize(self.cluster_embeddings, dim=-1),
        clip_feat_normalized
    ).reshape(-1, *clip_feat_normalized.shape[1:3])
    ```
    
    $$
    M_{IT}(x, y, n) 
    = \frac{f_I(x,y) \cdot f_T(n)}
           {\|f_I(x,y)\| \,\|f_T(n)\|}.
    $$
    
2. **Integrating masks into CLIP features**
    
    ![{091959FF-B093-448F-9B6C-DEEF7773CC30}.png](https://github.com/user-attachments/assets/34e235d8-1062-44aa-81f1-7d2066dd539a)
    
    - semantic class T에는 접근 가능하지 않고, **SAM/DINO로 도출된 unlabeled mask M** ∈ R^(HxWxN) (N=이미지 I당 마스크의 개수) 만 존재함
    - 마스크와 CLIP feature을 합치기 위해 CLIP 이미지 인코더를 unlabeled mask를 **supervision**으로 하여 파인튜닝함
    - 이미지를 input으로 CLIP 이미지 인코더에서 이미지 feature 맵 /f_i를 도출하고 **mask pooling으로 per-mask CLIP feature를 얻음**
    - f_M = MaskPool(f_i, M), f_M ∈ R^**(N x d)** 이미지당 마스크개수 * 차원 수
    - 즉 여기서 **image-mask similarity map** M_IM ∈ R^**(H x W x N)**
    - 이것으로 binary mask loss L_mask를 얻을 수 있으며, 이미지 I와 unlabeled mask M 으로 CLIP 이미지 인코더를 파인튜닝함
    - 실제로는 M_IM (**image-mask similarity map**이 CLIP 이미지 인코더 f_I와 같은 해상도이므로, 디코더 D를 이용하여 M_IM의 해상도를 마스크 M의 해상도로 만들어줌 M = D(M)
    
    **→ 이미지-마스크 코사인 유사도 맵 (H x W x N)**
    
    **→ 유사도 맵을 디코더를 통해 예측된 마스크로 변환**
    
    $$
    M_{IM}(x, y, n) 
    = \frac{f_I(x,y) \cdot f_M(n)}
           {\|f_I(x,y)\| \,\|f_M(n)\|}.
    $$
    
    ```python
    mask_preds = self.head(cost_volume.unsqueeze(1))
    mask_preds = F.interpolate(mask_preds, size=self.mask_res, mode="bilinear").squeeze().reshape(
        -1,
        self.cluster_embeddings.shape[0],
        *self.mask_res
    )
    ```
    
3. **Semantic clustering of masks**
    - DINO나 SAM에서 생성된 마스크는 이미지를 너무 과도하게 segment하는 경향이 있음
    - CLIP으로 의미론적으로(semantically) 비슷한 부분을 clustering 하는 로직이 필요
    - Global clustering: 각 이미지나 iteration에 제한된 것이 아닌, **전체 훈련과정에서 share됨**
    - 이것은 pixel-level semantic label을 만드는 것과 비슷하지만, **pre-defined된 클래스가 없다**는 것이 다름
    - **Online clustering via learnable class prompts**
        
        ![{D2BFB419-CC5E-40A8-8FFD-80D0EAF8A34C}.png](https://github.com/user-attachments/assets/93f2a642-2b7b-4008-9336-f9ec131b91f1)
        
        - 클러스터링 마스크의 centroid로 CLIP 텍스트 feature을 이용
        - 각 클러스터는 class-specific learnable prompts로 정의되어 (ex- A photo of [C1]) CLIP 텍스트 인코더에 들어감
        - 기존의 프롬프트 러닝이 prefix(A photo of a)에 초점을 맞추는 것에 비해, ‘class’를 학습하려 하였음
        - 클러스터의 개수가 k개면, prompt token은 C ∈ R^(k x l x d_e)
            - l = 프롬프트의 토큰 길이, d_e = token embedding의 차원
        - **CLIP 텍스트 인코더** E_L은 **클래스 feature f_C= E_L(P*, C) ∈ R^(k x d)을 만듦**
            - P* = CLIP 텍스트 인코더의 fixed template = prefix (A photo of a {} in the scene.)
        - 클래스가 모든 이미지의 general semantics를 나타내길 원했으므로,
        - 미니배치 안에 있는 m개의 마스크를 k개의 클러스터링(같은 의미를 가진 그룹)에 고르게 분배하여, 각 클러스터가 충분한 마스크의 양을 얻도록 함.
            - 즉, 한 클러스터가 다양한 마스크를 학습하도록 하여 일반화된 의미를 학습함
        - 이미지-텍스트 유사도 (mask pool된 feature f_M ↔ 텍스트 클래스 feature)에 따라, assignment Q ∈ R ^ (k x m) 를 얻음
            - k = 클러스터의 개수, m = 마스크의 개수
        
        $$
        \max_{\mathbf{Q} \in \mathcal{Q}}
        \;\mathrm{Tr}\!\bigl(\mathbf{Q}^{\top}\,\mathbf{F}_M^{\top}\,\mathbf{f}_C\bigr)
        \;+\;\varepsilon \, H(\mathbf{Q})
        \quad\\
        \text{subject to}
        \quad
        \mathbf{Q} \in \mathbb{R}_{+}^{k \times m},
        \quad
        \mathbf{Q}^{\top}\mathbf{1}_{k} \;=\; \tfrac{1}{m}\,\mathbf{1}_{m},
        \quad
        \mathbf{Q}\,\mathbf{1}_{m} \;=\; \tfrac{1}{k}\,\mathbf{1}_{k}.
        $$
        
        - Tr (행렬 대각합), H(Q) 는 엔트로피, ε는 엔트로피 regularization을 스케일링함으로써 매핑의 smoothness를 결정
        - **균등 분배 전략**
            - 각 행(클러스터)과 각 열(마스크)의 합을 균일하게 맞추는 Equipartition 제약
            - 평균적으로 하나의 클러스터가 (m/k)개의 마스크를 할당받게  됨
            - 실제로는, u(클러스터), v(마스크)의 정규화 → Sinkhorn-Knopp algorithm
                
                $$
                \mathbf{Q} \;=\; \mathrm{diag}(\mathbf{u})
                \;\exp\!\Bigl(\frac{\mathbf{F}_M^\top\,\mathbf{f}_C}{\varepsilon}\Bigr)
                \;\mathrm{diag}(\mathbf{v})
                $$
                
            
            ```python
               @torch.no_grad()
                def distributed_sinkhorn(self, logits, n_iters=3):
                    Q = logits.softmax(dim=-1).T 
                    B = logits.shape[1] # number of samples to assign
                    K = logits.shape[0] # how many prototypes
                    
                    # make the matrix sums to 1
                    sum_Q = torch.sum(Q)
                    if dist.is_initialized():
                        dist.all_reduce(sum_Q)
                        
                    B = sum_Q
                    Q /= sum_Q
            
                    for it in range(n_iters):
                        # normalize each row: total weight per prototype must be 1/K
                        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                        if dist.is_initialized():
                            dist.all_reduce(sum_of_rows)
                        Q /= sum_of_rows
                        Q /= K
            
                        # normalize each column: total weight per sample must be 1/B
                        Q /= torch.sum(Q, dim=0, keepdim=True)
                        Q /= B
            
                    Q *= B # the colomns must sum to 1 so that Q is an assignment
                    
                    return Q.t()
            ```
            
            ```python
            # compute mask features
            mask_feats = torch.einsum("hwc, nhw -> nc", teacher_feats[i], downsized_masks / downsized_masks.sum(dim=[1, 2], keepdim=True))
            similarity_map = F.normalize(mask_feats, dim=-1) @ F.normalize(self.cluster_embeddings, dim=-1).T # image-cluster similarity map
            ```
            
    - 이것으로, **이미지-클래스 코사인 유사도**를 구할 수 있음 (assigned class feature ↔ image feature)
        
        $$
        M_{IC}(x, y, i) 
        = \frac{f_I(x,y) \cdot f_C(i)}
               {\|f_I(x,y)\| \,\|f_C(i)\|}.
        $$
        
        - Assign된 f_M과 f_C(i) (f_C의 i번째 클래스 feature)에 대한 예측 마스크를 구할 수 있으며, GT 마스크 M 또한 argmax Q에 의해 클러스터됨
        
        ```python
        logits = torch.cat(logits, dim=0)
                assignment = self.distributed_sinkhorn(logits)
                hard_assignment = assignment.argmax(dim=-1)
                
                down_masks = torch.cat(down_masks, dim=0)
                
                # merge GT masks with respect to assigned clusters
                for i, n_mask in enumerate(num_masks):
                    assign = hard_assignment[num_masks[:i].sum():num_masks[:i].sum() + num_masks[i]]
                    mask = down_masks[num_masks[:i].sum():num_masks[:i].sum() + num_masks[i]]
                    target_label[i, assign] += mask
                    target_label[i].index_add_(0, assign, mask)
        ```
        
    - **Momentum 인코더**
        - 학습 시 instability와 forgetting problem을 위해 모멘텀 업데이트를 진행
        - f_M = MaskPool(E_V (I), M) , 여기서 E_V는 CLIP 이미지 인코더의 **모멘텀 인코더**임
        
        ```python
        def momentum_update(self, student, teacher, alpha=0.999):
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(alpha).add_(1 - alpha, param_s.data) 
        ```
        

### 4. Experiments

- Learned prompt token
    - 랜덤하게 초기화된 텍스트 프롬프트를 학습하여, 이것이 실제 클래스 이름과 유사한 위치로 분포(다양한 의미를 학습)하였음
    - 그러나 여전히 GT와 큰 차이를 보이는 것들이 보임
    - 즉 class prompt에 대한 연구 필요함

### 5. Conclusion & My thinking

Q. Momentum 인코더의 효과?
