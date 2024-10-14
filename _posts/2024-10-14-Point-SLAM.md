---
layout: post
title: Point-SLAM
tags: SLAM, Dense, Indirect, DL, RGBD
published: true
math: true
date: 2024-10-14 17:49 +0900
---

# Point-SLAM: Dense Neural Point Cloud-based SLAM


[https://www.youtube.com/watch?v=QFjtL8XTxlU](https://www.youtube.com/watch?v=QFjtL8XTxlU)

[https://github.com/eriksandstroem/Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)

### **Ⅰ. 연구의 배경**

- **Scene Representation**
    - **grid 기반**
        - dense grid, hierarchial octrees, voxel hashing
        - 장점: 이웃 탐색(neighborhood look up)과 컨텍스트 집계(context aggregation)기 쉽다.
        - 한계: 해상도가 먼저 정의되어야 하며, 중간에 바뀔 수 없다. 즉, 메모리 낭비가 크다.
    - **point 기반**
        - 장점: point의 밀도가 미리 정의될 필요가 없다. 또한, surface에 집중되어 free space를 모델링할 필요가 없으므로 메모리 낭비가 적다.
        - 한계: 이웃 탐색이 어렵다. (point간의 connectivity가 부족하다.)
        - 이웃 탐색의 해결책으로는 projection을 통해 3D탐색→2D탐색으로 바꾸는 것과 **grid에 point를 등록하는 것이 있다.**
    - **network 기반**
        - 장점: 연속적이고 압축적인 형태로, 높은 퀄리티의 매핑을 수행할 수 있다.
        - 한계: 지역적 업데이트를 허용하지 않는다. 런타임에서 network의 용량을 확장할 수 없다. (이는 forgetting problem을 일으킨다.)
        - 즉, **neural point feature을 3D 공간에 고정시키는 방식**으로, 지역적 업데이트와 확장을 허용한다.
    - 그 외의 기법
        - parameterized surface elements: 유연한 형상을 표현하기 어렵다.
        - axis aligned feature planes:: 과도하게 압축되어 여러 surface를 가진 경우 표현이 어렵다.


### **Ⅱ. Neural Point Cloud**

- **neural points**
    - **위치 p (3)**
    - **geometric/color feature descriptor f_g (32) and f_c (32)**
        
        $$
        P = \{(p_i, f_i^g, f_i^c)|i = 1, ..., N\}
        $$
        
- **point 추가 전략**
    - Input: 카메라 포즈, RGB-D 이미지
    - point sampling
        - image plane에서 X개 픽셀을 uniform sample한다.
        - 가장 큰 color gradient를 가진 5Y개의 픽셀 중 Y개 픽셀을 uniform sample한다.
    - 3D 탐색
        - 역투영 후 반경=r에서 이웃을 탐색한다.
        - 이웃이 없으면, ray 상에 **3개의 nerual point를 추가**한다. (깊이=D, (1-p)D, (1+p)D / p=(0,1) →깊이맵 노이즈 때문)
        - 이웃이 있으면 point를 추가하지 않는다.
    - 업데이트
        - feature vector는 normally distribution으로 초기화된다.
        - 깊이맵의 noise를 고려해, 3개의 point는 depth에 따라 **update를 제한하는 대역**으로 존재한다.
        - 따라서 scene이 진행되어도, neural point cloud는 bounded points로 수렴할 수 있다.
        - voxel 기반 기법과 달리 scene의 bound를 먼저 정의할 필요가 없다.
- **동적 밀도 (dynamic point density)**
    - fine/coarse detail을 효과적으로 모델링 가능하다.
    - **color gradient에 따라 반경=r을 동적으로 조정**한다. (linear mapping)
    
    $$
    r(u, v) = 
    \begin{cases} 
    r_1 & \text{if } |\nabla I(u, v)| \geq g_u \\
    \beta_1 |\nabla I(u, v)| + \beta_2 & \text{if } g_l \leq |\nabla I(u, v)| < g_u \\
    r_u & \text{if } |\nabla I(u, v)| < g_l 
    \end{cases}
    $$
    
    - pixel location (u,v) → gradient ∇I(u, v)
    - gradient가 크면 반경이 작고(lower), 작으면 반경이 크다(upper).
        - 변화율이 큰 영역은 **선명하게 경계가 나누어지는 디테일한 이미지로**,  이웃 탐색 반경을 작게 함으로써, 많은 구(sphere?)가 만들어지고, 디테일을 보존한다.
        - 변화율이 작은 영역은 **세부 사항이 적은 평탄한 이미지**로, 이웃 탐색 반경을 크게 함으로써, 적은 구를 만들 수 있고 이는 효율성을 크게 한다.


### **Ⅲ. 렌더링**

- **uniform sampling points near surface**
    
    $$
    x_i = O + z_id, \ i ∈ \{1, ..., M\}
    $$
    
    - 픽셀의 깊이 D 주변, (1-p)D와 (1+p)D 사이에서 5개의 point를 uniform sample한다.
    - free space를 sample하지 않고 surface 주변에서만 sample할 수 있다는 장점을 가진다.
- 깊이 값이 없는 경우
    - *깊이맵 값이 없는 픽셀*에 대해서 30cm부터 1.2D_max (최대 깊이) 까지 ray를 25개 sample (=ray marching) 한다.
    - 이는 **hole filling**의 효과를 준다. (그러나 너무 큰 hole에는 효과 없음)
- **MLP (decoder)**
    
    $$
    o_i = h(x_i, P^g(x_i))
    \\
    c_i = g_ξ(x_i, P^c(xi))
    
    $$
    
    - point + geometry feature (from point) ⇒ **occupancy**
    - point + color feature (from point) ⇒ **color**
    - h는 미리 학습된 변하지 않는 MLP이며,
    - g는 trainable parameter인 ξ을 가지는 MLP이다.
    - h와 g의 구조는 같으며, 이후 **Gaussian positional encoding**을 적용한다.
- **feature**
    - geometry feature
        - 각 point에서 2r만큼의 크기에서 적어도 2개의 이웃을 찾으며, 그 미만이면 **zero occupancy**이다.
        - 가장 가까운  **8개의 이웃**을 찾고 inverse square distance weight를 통해 geometric feature을 계산한다.
        - 즉, 가까울수록 weight가 커지며, geometry feature의 영향이 커진다.
    
    $$
    P^g(x_i) = \sum_k\frac{w_k}{\sum_k w_k}f_k^g \\ with \ w_k =\frac{1}{||pk − xi||^2}
    $$
    
    - color feature
        - point-nerf에 영향받았다.
        - F는 one-layer MLP이며, 128개의 neuron과 softplus activation을 가진다.
        - point vector (p-x), 즉 각 이웃으로부터 포인트를 가리키는 벡터에 **Gaussian positional encoding**을 적용한다. → ?
        
        $$
        f^c_{k,x_i} = F_θ(f^c_k, p_k − x_i),\\
        $$
        
        $$
        P^c(x_i) = \sum_k\frac{w_k}{\sum_k w_k}f^c_{k,x_i}
        $$
        
- **volume rendering**
    - 전반적인 기법은 nerf와 동일하다.
    - weight function (occupancy)
        
        $$
        α_i = o_{\textbf{p}_i}\prod_{j=1}^{i-1}(1 − o_{\textbf{p}_j})
        $$
        
    - **rendered depth, rendered color (predicted)**
        
        $$
        \hat{D} =\sum^N_{i=1}α_iz_i,\\
        \hat{I} =\sum^N_{i=1}α_i\textbf{c}_i
        $$
        
    - **depth variance** (분산)
        
        $$
        \hat{S}_D = \sum^N_{i=1}α_i(\hat{D }− z_i)^2 
        $$
    
        

### **Ⅳ. 매핑/트래킹**

- 매핑
    - **rendering loss(=mapping loss)** 계산을 통해 각 포인트에서 **geometric/color feature을 최적화한다.** (decoder을 최적화, interpolation decoder F도 포함)
    - L1 loss를 사용하였다. (→**why?** outlier에 덜 민감하기 때문?)
        
        $$
        L_{map} = \sum^M_{m=1}|D_m − \hat{D}_m|_1 + λ_m|I_m − \hat{I}_m|_1
        $$
        
    - 처음에는 depth loss만을 이용하여 최적화하지만, 나머지 60퍼센트의 iteration에서는 color loss도 추가한다.
        - 왜냐하면 처음에는 color가 매우 부정확하기 때문이다. (추측)
    - **키프레임 데이터베이스**를 만들어 mapping loss를 정규화한다.
        - NICE-SLAM에 영향받았다.
        - 키프레임과 현재 프레임의 view frustum이 많이 겹치면, (=카메라가 비슷한 방향에서 바라봄) 키프레임으로부터 픽셀을 sample한다.
- 트래킹
    - **카메라 extrinsic (R, t)를 각 프레임마다 최적화**한다.
    - 한 프레임에서 **M_t** 픽셀을 샘플한다.
    - **simple constant speed assumption**을 통해 i-2와 i-1번째 프레임의 포즈 변환만큼 i-1와 i번째(현재) 프레임의 포즈 변환을 초기화한다.
    - tracking loss는 color loss와 depth loss+표준편차를 결합한다.
        - 이는 iMAP에서 따온 것으로 보인다. (추후 NICE-SLAM에서도 적용됨)
        - 사물의 경계는 보는 각도에 따라 깊이의 편차가 매우 크게 측정된다. 즉, loss가 큰데, 이를 해결하기 위해 정규화를 적용한 것이다.
        
        > The geometric loss measures the depth difference and uses the depth variance as a **normalisation factor,** **down-weighting the loss in uncertain regions such as object borders:**
        > 
        
        $$
        L_{\text{track}} = \sum_{m=1}^{M_t} \left(\frac{|D_m - \tilde{D}_m|}{\sqrt{\hat{S}_D}} + \lambda_t |I_m - \hat{I}_m|\right)
        $$
        
- **Exposure Compensation** (선처리)
    - 프레임들 간 큰 노출의 차이가 있는 경우 (color의 차이가 큼), 픽셀 간 색상 차이를 줄인다.
    - per-image latent vector→ exposure MLP Gϕ → affine transformation
    - tracking / mapping loss를 게산하기 전에 I^을 변화시킨다.
    - **MLP Gϕ**
        - 1 hidden layer, 128 neuron, softplus activation
        - input: 8D latent vector
        - output: 12D, (3x3 affine matrix + 3x1 translation)
    - 다른 potential solution이 있을 수 있음
        - exposure mapping을 딥러닝에 맡긴 점이 흥미로웠다..


### **V. 결론**

![image.png](https://file.notion.so/f/f/ba6daae0-3a94-4ebe-93cd-85879b4d7406/ad629f90-b232-43c8-b0ac-8e271c47aeb8/image.png?table=block&id=11fb5d10-e488-80e0-8f37-f87ea3e55cc6&spaceId=ba6daae0-3a94-4ebe-93cd-85879b4d7406&expirationTimestamp=1729036800000&signature=EQwxkeJiQMydweua_dqNfXtyKRiM_9fgArTeThjZTa8&downloadName=image.png)

- synthetic dataset인 Replica에서는 월등한 결과를 보였지만, TUM-RGBD 이나 ScanNet과 같이 실제 깊이 카메라로 촬영하여 noise가 발생하는 경우에는 tracking 성능이 떨어진다.
- 모션 블러나 반사(specularity)에 취약하다.
- 키프레임 선택 전략, 반경(radius)의 선택을 위한 color gradient의 upper/lower bound가 heuristic하게 선정되었다.
- 내 생각
    - 이 논문의 contribution이 point sampling / point 추가 전략 (+이웃 탐색 전략)이라고 생각한다.
    - 이 전략을 좀 더 발전시킬 수 있지 않을까 생각이 든다. 3D탐색→2D탐색도 괜찮은 생각인 것 같은데…
    
    
    - 혹은 꼭 point여야 하나?는 생각이 든다. 이것을 gaussian으로 바꾼다면..?
