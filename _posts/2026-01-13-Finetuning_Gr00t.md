---
layout: post
title: Finetuning Gr00t N1.6 with LeRobot Dataset
tags: VLA
published: true
math: true
date: 2026-01-13 09:00 +0900
---


# Isaac GR00T N1.6을 LeRobot SO-101 Arm으로 파인튜닝하기

## **참고자료**

- **GR00T N1.5 버전 Manual**

[https://velog.io/@choonsik_mom/Leisaac-LeRobot-Gr00t-IsaacSim%EC%9C%BC%EB%A1%9C-%EC%9E%85%EB%AC%B8%ED%95%98%EB%8A%94-VLA-Finetuning](https://velog.io/@choonsik_mom/Leisaac-LeRobot-Gr00t-IsaacSim%EC%9C%BC%EB%A1%9C-%EC%9E%85%EB%AC%B8%ED%95%98%EB%8A%94-VLA-Finetuning)

[https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)

- **GR00T N1.6 버전 Manual**

[https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/SO100](https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/SO100)

## 사건

### LeIsaac 발견

LeIsaac = LeRobot + GR00T + Isaac Sim을 연결한다고 해서 실행해보고자 하였다.

(GR00T 에서 LeRobot SO-101 등 로봇 파인튜닝 + Isaac Sim에서 eval 볼 수 있음)

### 버전이 안 맞는 문제

블로그에서 본 대로 따라하려 하였더니, 스크립트가 없었다. (`scripts/gr00t_finetune.py`)

알고보니 Gr00t가 N1.5버전이었다.

나는 N1.6으로 파인튜닝 하고 싶었기 때문에 방법을 찾아보았다.

`scripts/gr00t_finetune.py` 대신 `gr00t/experiment/launch_finetune.py` 가 쓰인다는 사실을 알아냈다. 그러나 문제가 있었다.

1. Lerobot dataset이 v3이어서, `episodes.jsonl`이 존재하지 않음
2. `ValueError: Default process group has not been initialized, please make sure to call init_process_group.` (분산학습 문제)
3. `line 359, in _load_video_data original_key = self.modality_meta["video"][image_key].get(KeyError: 'front')` (front, wrist가 없는 문제)
4. uv / conda env 관련… 헷갈리는 부분
    
    

## 해결

1. Lerobot dataset이 v3이어서, `episodes.jsonl`이 존재하지 않음
    - 이 경우 해결책은 다음과 같다.  [https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/lerobot_conversion/README.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/lerobot_conversion/README.md)
    - 이 스크립트를 이용해 LeRobot v3에서 v2로 바꾼 후 실행하면 된다.
    - `python convert_v3_to_v2.py --repo-id <hf_id/hf_repo> --local-dir <download_path>`
2. `ValueError: Default process group has not been initialized, please make sure to call init_process_group.` (분산학습 문제)
    - 이 경우 제미나이가 알려준 해결책은 일단 torchrun 부분을 주석해제 하는 것이다.
    - 그래도 문제가 생기는데, 그것은 바로
    - 내 실행코드는 다음과 같다.
        
        ```bash
        torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
            gr00t/experiment/launch_finetune.py \
            --base_model_path nvidia/GR00T-N1.6-3B \
            --dataset_path  demo_data/finish_sandwich \
            --modality_config_path examples/SO100/so100_config.py \
            --embodiment_tag NEW_EMBODIMENT \
            --num_gpus $NUM_GPUS \
            --output_dir ./output/so100_finetune \
            --save_steps 1000 \
            --save_total_limit 5 \
            --max_steps 10000 \
            --warmup_ratio 0.05 \
            --weight_decay 1e-5 \
            --learning_rate 1e-4 \
            --use_wandb \
            --global_batch_size 32 \
            --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
            --dataloader_num_workers 4
        ```
        
3. `line 359, in _load_video_data original_key = self.modality_meta["video"][image_key].get(KeyError: 'front')` (front, wrist가 없는 문제)
    
    `examples/SO100/modality.json`에서 `<dataset_name>/meta/modality.json` 으로 복사해왔는데, 원래 example 상에서는 실제 로봇 모달리티이기 때문에 video부분에 webcam만 있다. 근데 데이터셋 상에서는 wrist와 front 카메라가 있는 것이다. 
    
    그렇기 때문에 다음과 같이 추가해줘야 한다.
    
    ```bash
     "video": {
            "webcam": {
                "original_key": "observation.images.webcam"
            },
            # 추가
            "front": {
                "original_key": "observation.images.front"
            },
            "wrist": {
                "original_key": "observation.images.wrist"
            }
    ```
    
4. uv / conda env 관련… 헷갈리는 부분
이건 내가 까먹은 게 잘못인데, 처음에 환경을용해서 실행해야 할지 몰랐다.
    - LeRobot은 학습편의를 위해 client(robot) ↔ server(VLA model)을 나눈다.
    - 그니까 학습(finetune)할때는 uv로 설정한 gr00t 를 위한 환경으로 실행하면 됨
    - 내가 isaacsim은 conda로 했으니까 나중에 Leisaac을 eval할때는 server를 켜놓고 따로 isaacsim을 실행하면 되는 것이다. (아마도)
    - Isaac-GR00T closed loop eval 의 경우 는 다음과 같다. 난 로봇이 없으니까 할 수 없다.
    
    [https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/SO100#closed-loop-evaluation](https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/SO100#closed-loop-evaluation)
    

## 결과

![image.png](/assets/Robot/gr00t.png)

일단 학습 자체는 되는 듯하다…