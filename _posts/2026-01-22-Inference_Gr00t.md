---
layout: post
title: Inference Gr00t N1.6 with Isaac Sim
tags: VLA
published: true
math: true
date: 2026-01-22 09:00 +0900
---


# Isaac GR00T N1.6을 LeRobot SO-101 Arm & IsaacSim으로 Inference하기

Isaac GR00T N1.6을 LeRobot SO-101 Arm & IsaacSim으로 Inference하기

파인튜닝만 하면 Inference 는 아주 간단하다. 먼저 evaluation을 실시한다.

- **eval.sh**
    
    ```bash
    python gr00t/eval/open_loop_eval.py \
       --save-plot-path ./eval_plots \
       --embodiment_tag NEW_EMBODIMENT \
       --model_path <CHEKPOINT_PATH>  \
       --dataset_path demo_data/leisaac-pick-orange \
       --modality_keys single_arm gripper
    ```
    

`uv run bash eval.sh` 로 실행하면 된다.

eval 결과는 다음과 같이 plot으로 나온다.

![/assets/Robot/plot_eval.png](/assets/Robot/plot_eval.png)

그런데, 성공률이 처참하다.

1. GR00T 서버 열기
- **run_server.sh**
    
    ```bash
    python gr00t/eval/run_gr00t_server.py \
    --model_path <CHEKPOINT_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --dataset_path demo_data/leisaac-pick-orange \
    --modality_config_path examples/SO100/so100_config.py \
    --execution_horizon 16
    ```
    
- `uv run bash run_server.sh` 로 실행하면 된다.

1. LeIsaac 에서 Isaac Sim을 실행하여 Inference하기
- **inference.sh**
    
    ```bash
    python scripts/evaluation/policy_inference.py \
        --task=LeIsaac-SO101-PickOrange-v0 \
        --eval_rounds=10 \
        --policy_type=gr00tn1.6 \
        --policy_host=localhost \
        --policy_port=5555 \
        --policy_timeout_ms=5000 \
        --policy_action_horizon=16 \
        --policy_language_instruction="Pick up the orange and place it on the plate" \
        --device=cuda \
        --enable_cameras
    ```
    
- Leisaac env 에서 `bash inference.sh` 으로 실행하면 된다.

1. 결과
    
    상당히 발발거리면서 임무 완수를 잘 해내지 못하는 것을 볼 수 있다. 성공률 0.1인걸 봐서 뭔가 파인튜닝 과정에 문제가 있는 것 같다.
    
    ```bash
    [Evaluation] Evaluating episode 1...
    [Evaluation] Episode 1 timed out!
    [Evaluation] now success rate: 0.0  [0/1]
    [Evaluation] Evaluating episode 2...
    [Evaluation] Episode 2 timed out!
    [Evaluation] now success rate: 0.0  [0/2]
    [Evaluation] Evaluating episode 3...
    [Evaluation] Episode 3 timed out!
    [Evaluation] now success rate: 0.0  [0/3]
    [Evaluation] Evaluating episode 4...
    [Evaluation] Episode 4 timed out!
    [Evaluation] now success rate: 0.0  [0/4]
    [Evaluation] Evaluating episode 5...
    [Evaluation] Episode 5 is successful!
    [Evaluation] now success rate: 0.2  [1/5]
    [Evaluation] Evaluating episode 6...
    [Evaluation] Episode 6 timed out!
    [Evaluation] now success rate: 0.16666666666666666  [1/6]
    [Evaluation] Evaluating episode 7...
    [Evaluation] Episode 7 timed out!
    [Evaluation] now success rate: 0.14285714285714285  [1/7]
    [Evaluation] Evaluating episode 8...
    [Evaluation] Episode 8 timed out!
    [Evaluation] now success rate: 0.125  [1/8]
    [Evaluation] Evaluating episode 9...
    [Evaluation] Episode 9 timed out!
    [Evaluation] now success rate: 0.1111111111111111  [1/9]
    [Evaluation] Evaluating episode 10...
    [Evaluation] Episode 10 timed out!
    [Evaluation] now success rate: 0.1  [1/10]
    [Evaluation] Final success rate: 0.100  [1/10]
    ```
    
    - 에피소드가 너무 적기 때문으로 추정된다..
    - 혹은 세번째 사진처럼, 물리적 한계에 도달한 경우도 있었다. (잡은 것처럼 보이지만 실제는 너무 멀리 있는 것임/낀 경우도 있음)
    
    ![/assets/Robot/isaacsim1.jpg](/assets/Robot/isaacsim1.jpg)
    
    ![/assets/Robot/isaacsim2.png](/assets/Robot/isaacsim2.png)
    
    ![/assets/Robot/isaacsim3.png](/assets/Robot/isaacsim3.png)
    

다음에는 실제로 데이터셋을 구성해 학습시키고, 동일한 방식으로 실행 해보려 한다.

SO-100 ARM이 실제로 온다면.. 그땐 텔레오퍼레이션도 해볼 수 있겠지….