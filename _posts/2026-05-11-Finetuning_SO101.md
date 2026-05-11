---
layout: post
title: SO-101 Arm 텔레오퍼레이션으로 데이터 생성하고 파인튜닝하기
tags: VLA, Robotics
published: true
math: true
date: 2026-05-11 09:00 +0900
---


### SO-101 Arm 텔레오퍼레이션으로 데이터 생성하고 파인튜닝하기

해본 지는 오래됐는데, 그동안 시간이 하나도 없어서 작성을 못헸다.

### 참고 자료

https://huggingface.co/docs/lerobot/so101

https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning

https://cory619.tistory.com/25

https://wikidocs.net/325771

### 환경 설정

일단, 내가 가지고 있었던 SO-101 Arm은 이미 완벽하게 조립이 된 상태였다. 따라서, motor configuration & calibration으로 시작하는 teleoperation 부터 작성해보도록 하겠다.

- SO-101 Arm은 총 두 가지가 있다. Leader(내가 잡는 것), Follower(Leader의 움직임을 따라가는 것)
- 두 로봇에 대해 각각 motor configuration과 calibration을 실시해야 한다.
- **전원 어댑터 연결 시, Follower 보드는 12V 이고 Leader 보드는 5V이므로 주의해야 한다.**

#### motor configuration

사실 이건 원래 조립할 때 하는 거지만, 나는 이미 조립된 것을 사용하여야 했어서 지금 단계에 수행했다.

1. USB 권한 부여
    
    USB 포트의 이름을 먼저 식별한다.
    
    ```python
    lerobot-find-port
    ```
    
    출력 예시: 
    
    ```python
    Finding all available ports for the MotorBus.
    ['/dev/ttyACM0', '/dev/ttyACM1']
    Remove the usb cable from your MotorsBus and press Enter when done.
    [...Disconnect corresponding leader or follower arm and press Enter...]
    
    The port of this MotorsBus is /dev/ttyACM1
    Reconnect the USB cable.
    ```
    
    네 경우, `dev/ttyACM0`와 `dev/ttyACM1`이였다. (이게 항상 고정된 것은 아니고 먼저 꽂는 게 0, 나중에 꽂는 게 1이다.)
    
    그런데, 그냥 하면 permission denied가 되므로,  `sudo chmod 666 /dev/ttyACM0`와 `sudo chmod 666 /dev/ttyACM1` 을 통해 권한을 부여해주면 된다.
    
    주의할 점은, Follower은 robot 이지만 Leader은 teleop인 것이다. 앞으로도 똑같으므로 헷갈리지 않도록 한다.
    
2. (이미 조립되어 있는 경우) 각 모터에서 선을 하나씩 빼서 컨트롤러 보드에 연결하여야 한다. 
    
    아까 로봇을 USB 케이블로 연결했으니, 각 모터의 이름을 식별하도록 설정해야 한다.
    
    ```python
    lerobot-setup-motors \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0
    ```
    
    ```python
    lerobot-setup-motors \
        --teleop.type=so101_leader \
        --teleop.port=/dev/ttyACM1
    ```
    
    ```python
    Connect the controller board to the 'gripper' motor only and press enter.
    ```
    
    **맨 위(그리퍼)에서부터 가장 마지막(베이스)까지 타고 내려오면서 하나씩 제거해준 후, 컨트롤 보드에 연결해서 엔터를 치면 된다.**
    
    처음에는 헷갈리지만 직렬 연결이라고 생각하면 편하다. 이 모터는 ST3215 Serial Bus Servo 인데, 각 모터마다 2개의 커넥터가 있어 서로 직렬로 연결되는 것이다.
    
    실제로, 이런식으로 연결된다. Leader의 경우, gripper가 Follower와는 다르게 생겼지만 어쨌든 손가락을 넣어 그리퍼를 표현할 수 있게 해놓았으므로 모터를 잘 찾으면 된다.
    
    ![GPT가 그려준 사진](/assets/post_260511/f1b762a5-9af6-4f7c-8b04-2e1d28ddbc13.png)
    
    GPT가 그려준 사진
    
    |  Joint 1 |  Joint 2 |  Joint 3 |  Joint 4 |  Joint 5 |  Joint 6 |
    | --- | --- | --- | --- | --- | --- |
    | 베이스 | 숄더 | 엘보 | 손목 굽힘 | 손목 회전 | 그리퍼 |
    | Board - 2 | 1 - 3 | 2 - 4 | 3 - 5 | 4 - 6 | 5 - X |
3. 주의 사항 - 잘 안 되는 경우
    - 전원 어댑터를 잘못 연결한 경우
    - 선을 헷갈려서 모터를 잘못 입력한 경우 - 뭔가 문제가 있다면 모터 연결부터 다시 해보는 것을 추천

#### calibration

모터를 잘 입력했다면 이제 캘리브레이션을 하는데, Leader와 Follower의 위치값을 맞추어 같은 양만큼 움직이게 조절해주는 것이다.

1. 클램프 고정
    
    이를 위해 중요한 것은 클램프로 로봇을 반드시 고정시켜줘야 하는데, 이것이 없다면 가동 범위만큼 움직일 수 없으므로 필수적이다. 3D 프린팅으로 출력한 클램프의 성능은 매우 좋지 않으므로, 다른 클램프를 구하는 것을 추천한다.
    
2. 캘리브레이션 명령어 입력
    
    ```python
    lerobot-calibrate \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_awesome_follower_arm
    ```
    
    ```python
    lerobot-calibrate \
        --teleop.type=so101_leader \
        --teleop.port=/dev/ttyACM1 \
        --teleop.id=my_awesome_leader_arm
    ```
    
    명령어를 입력하면
    
    ```python
    Move my_awesome_leader_arm SO101Leader to the middle of its range of motion and press ENTER....
    ```
    
    ‘Middle’ position이란 아래처럼, ㄱ자로 로봇을 놓아주면 된다.
    
    ![image.png](/assets/post_260511/ee8b8720-053c-41fd-ac97-3d9c8a18c052.png)
    
    ```python
    Move all joints sequentially through their entire ranges of motion.
    Recording positions. Press ENTER to stop...
    ```
    
    이렇게 나오는데, 이리저리 돌리고 굽혀가며 최대 / 최소 가동범위를 대충 맞춰주면 된다. 완벽히 같을 필요는 없고, 아래와 같은 정도의 값이 나오면 되는 듯하다.
    
    ![image.png](/assets/post_260511/image.png)
    
3. 파일 저장 확인
    
    ```python
    Calibration saved to ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json
    ```
    

### 데이터 수집

#### teleoperation

teleoperate는 데이터 수집 없이 joint값만 확인하는 스크립트이다.

```python
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="front:
        type: opencv
        index_or_path: 0
        width: 640
        height: 480
        fps: 30" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

지금부터는 상당히 쉬운데, lerobot에서 워낙 잘 스크립트를 만들어 놓았기 때문에 카메라만 연결해서 하면 된다. 나는 웹캠이 없었으므로 그냥 다이소에서 카메라를 샀지만, 좋은 카메라를 사는 것이 나은 것 같다.

카메라 포트만 잘 확인해서 index를 입력해주면 된다.

![KakaoTalk_20260511_175137570.gif](/assets/post_260511/KakaoTalk_20260511_175137570.gif)

#### record

record는 이미테이션 러닝을 위해 데이터를 수집하고 저장하는 역할이다.

```python
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --robot.cameras="front:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30" \
  --dataset.repo_id=eainx/pick_up_dorami_new \
  --dataset.single_task="Pick up the doll and put it on the yellow sticky note." \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.push_to_hub=true
```

**주요 Record 옵션 설명**

1. 로봇 설정
    - `-robot.type=so101_follower`: 팔로워 로봇 타입
    - `-robot.port=/dev/so101_follower`: 팔로워 로봇 포트
    - `-robot.id=follower`: 팔로워 로봇 고유 ID
2. 텔레오퍼레이션 설정
    - `-teleop.type=so101_leader`: 리더 로봇 타입
    - `-teleop.port=/dev/so101_leader`: 리더 로봇 포트
    - `-teleop.id=leader`: 리더 로봇 고유 ID
3. 데이터셋 설정
    - `-dataset.repo_id`: HuggingFace Hub 데이터셋 이름 (`username/dataset_name`)
    - `-dataset.single_task`: 작업에 대한 명확한 설명
    - `-dataset.fps`: 데이터 수집 주파수 (기본값: 30Hz)
    - `-dataset.num_episodes=5`: 수집할 에피소드 수
    - `-dataset.episode_time_s=15`: 각 에피소드 녹화 시간 (기본값: 60초)
    - `-dataset.reset_time_s=3`: 에피소드 간 리셋 시간 (기본값: 60초)
4. 추가 옵션
    - `-display_data=true`: 실시간 데이터 시각화 (기본값: false)
    - `-dataset.video=true`: 비디오 인코딩 활성화 (기본값: true)
    - `-dataset.push_to_hub=true`: HuggingFace Hub에 자동 업로드 (기본값: true)

상당히 재미있는데, episode_time_s 이 지나면 녹화가 끝나고 바로 리셋이 된다. 리셋 시간도 정해져 있어서, 이 시간 동안 environment를 원상복구 시켜야 한다.

하는 도중 선이 끊기거나 하면 바로 문제가 생겨서 클램프로 로봇을 잘 고정시키고 진행해줘야 한다.

데이터셋은 huggingface `eainx/pick_up_dorami`에 올려두었다. 데이터셋 품질이 안 좋아서 활용도는 없다.

### 파인튜닝

그래도 데이터 수집을 했으니 smolVLA을 파인튜닝해보았다. (결과는 좋지 않다..)

Train 명령어는 다음과 같다. 8000 step만 진행했다.

```python
lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=eainx/pick_up_dorami \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --output_dir=outputs/smolvla_pick_up_dorami \
    --batch_size=16 \
    --steps=8000 \
    --save_freq=1000 \
    --policy.push_to_hub=true \
    --policy.repo_id=smolvla_pick_up_dorami \
    --wandb.enable=true \
    --policy.device=cuda
```

Inference 명령어는 다음과 같다. (Follower에만 policy를 입력하므로, Leader은 필요 없다.)

```python
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="front:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30" \
  --dataset.single_task="Pick up the doll and put it on the yellow sticky note." \
  --dataset.repo_id=eainx/eval_pick_up_dorami_test \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --policy.repo_id=eainx/smolvla_pick_up_dorami
```

![KakaoTalk_20260511_175123898.gif](/assets/post_260511/KakaoTalk_20260511_175123898.gif)