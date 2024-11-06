---
layout: post
title: 3D Gaussian Splatting 코드 분석
tags: 3DReconstruction, Rendering, ViewSynthesis, Gaussian
published: true
math: true
date: 2024-11-06 21:00 +0900
---

# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

# 코드 분석

## 실행 코드

1. **convert.py**
    
    ```python
    python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
    ```
    
    - input 이미지로 COLMAP을 실행함
    - 1/2, 1/4, 1/8 사이즈로 이미지를 압축함
2. **train.py**
    
    ```python
    python train.py -s <path to COLMAP or NeRF Synthetic dataset>
    ```
    
    	<details>
	<summary>Command Line Arguments for train.py</summary>
	<div>

        - -source_path / -s
        
        Path to the source directory containing a COLMAP or Synthetic NeRF data set.
        
        - -model_path / -m
        
        Path where the trained model should be stored (`output/<random>` by default).
        
        - -images / -i
        
        Alternative subdirectory for COLMAP images (`images` by default).
        
        - -eval
        
        Add this flag to use a MipNeRF360-style training/test split for evaluation.
        
        - -resolution / -r
        
        Specifies resolution of the loaded images before training. If provided `1, 2, 4` or `8`, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
        
        - -data_device
        
        Specifies where to put the source image data, `cuda` by default, recommended to use `cpu` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
        
        - -white_background / -w
        
        Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
        
        - -sh_degree
        
        Order of spherical harmonics to be used (no larger than 3). `3` by default.
        
        - -convert_SHs_python
        
        Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
        
        - -convert_cov3D_python
        
        Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
        
        - -debug
        
        Enables debug mode if you experience erros. If the rasterizer fails, a `dump` file is created that you may forward to us in an issue so we can take a look.
        
        - -debug_from
        
        Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
        
        - -iterations
        
        Number of total iterations to train for, `30_000` by default.
        
        - -ip
        
        IP to start GUI server on, `127.0.0.1` by default.
        
        - -port
        
        Port to use for GUI server, `6009` by default.
        
        - -test_iterations
        
        Space-separated iterations at which the training script computes L1 and PSNR over test set, `7000 30000` by default.
        
        - -save_iterations
        
        Space-separated iterations at which the training script saves the Gaussian model, `7000 30000 <iterations>` by default.
        
        - -checkpoint_iterations
        
        Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
        
        - -start_checkpoint
        
        Path to a saved checkpoint to continue training from.
        
        - -quiet
        
        Flag to omit any text written to standard out pipe.
        
        - -feature_lr
        
        Spherical harmonics features learning rate, `0.0025` by default.
        
        - -opacity_lr
        
        Opacity learning rate, `0.05` by default.
        
        - -scaling_lr
        
        Scaling learning rate, `0.005` by default.
        
        - -rotation_lr
        
        Rotation learning rate, `0.001` by default.
        
        - -position_lr_max_steps
        
        Number of steps (from 0) where position learning rate goes from `initial` to `final`. `30_000` by default.
        
        - -position_lr_init
        
        Initial 3D position learning rate, `0.00016` by default.
        
        - -position_lr_final
        
        Final 3D position learning rate, `0.0000016` by default.
        
        - -position_lr_delay_mult
        
        Position learning rate multiplier (cf. Plenoxels), `0.01` by default.
        
        - -densify_from_iter
        
        Iteration where densification starts, `500` by default.
        
        - -densify_until_iter
        
        Iteration where densification stops, `15_000` by default.
        
        - -densify_grad_threshold
        
        Limit that decides if points should be densified based on 2D position gradient, `0.0002` by default.
        
        - -densification_interval
        
        How frequently to densify, `100` (every 100 iterations) by default.
        
        - -opacity_reset_interval
        
        How frequently to reset opacity, `3_000` by default.
        
        - -lambda_dssim
        
        Influence of SSIM on total loss from 0 to 1, `0.2` by default.
        
        - -percent_dense
        
        Percentage of scene extent (0--1) a point must exceed to be forcibly densified, `0.01` by default.
        
        </div>
	</details>
    - 초기화
        - 가우시안 모델 생성
            
            ```python
            gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            ```
            
        - scene 생성
            
            ```python
            scene = Scene(dataset, gaussians)
            ```
            
        - exponential learning decay scheduling (position에 대해서)
            
            ```python
            depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
            ```
            
    - 학습
        - `opt.iterations`(30,000번)만큼 반복 학습
        - 1000번마다 SH의 레벨을 올림 (최대 3)
        - 랜덤하게 카메라의 뷰포인트 선택
            
            ```python
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            ```
            
        - 가우시안 렌더 생성
            - `image`: 렌더된 이미지
            - `viewspace_point_tensor`: 2D 이미지에서의 가우시안 중심점
            - `visibility_filter`: radii가 0보다 큰 경우를 filter함
            - `radii`: 3D 가우시안의 반경
            
            ```python
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            ```
            
        - loss 계산
            
            ```python
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            ```
            
        - Depth 정규화
            
            ```python
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
            
                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0
            ```
            
    - Densification
        - `opt.densify_from_iter`(500번) 부터 `opt.densify_until_iter`(15000번) 전까지 `opt.densification_interval`(100번)마다 densify & prune 수행
        - 가우시안 개수 조절을 위해 alpha(불투명도)를 `opt.opacity_reset_interval`(3000번)마다 0으로 조정
        
        ```python
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
        
        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()
        ```
        
3. **render.py**
    - train/test set을 렌더
    
    ```python
    python render.py -m <path to trained model>
    ```
    
4. **metrics.py**
    - SSIM / PSNR / LPIPS 계산
    
    ```python
    python metrics.py -m <path to trained model>
    ```
    
5. full_eval.py
    
    ```python
    python full_eval.py -m <directory with evaluation images>/garden ... --skip_training --skip_rendering
    ```
    

## scene

1. **__init__.py**
    - scene 클래스를 정의
        
        ```python
        class Scene:
            gaussians : GaussianModel
        		def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        		    self.model_path = args.model_path
        		    self.loaded_iter = None
        		    self.gaussians = gaussians
        ```
        
        - load iteration
            - render의 경우 이미 존재하는 모델의 iteration 횟수를 불러옴
            - `Default`의 경우 `None`으로 설정됨
                
                ```python
                if load_iteration:
                    if load_iteration == -1: # render
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                    else: # train
                        self.loaded_iter = load_iteration
                    print("Loading trained model at iteration {}".format(self.loaded_iter))
                ```
                
        - scene 설정 (Colmap/Blender)
            - `sceneLoadTypeCallbacks`를 통해 카메라 파라미터와 depth, 포인트 클라우드를 불러와 `scene_info` 를 정의
                
                ```python
                if os.path.exists(os.path.join(args.source_path, "sparse")):
                    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
                elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                    print("Found transforms_train.json file, assuming Blender data set!")
                    scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
                else:
                    assert False, "Could not recognize scene type!"
                ```
                
        - 카메라 리스트
            - self.loaded_iter=`None` 일 때 카메라 리스트를 불러옴 (train/test)
                
                ```python
                if not self.loaded_iter:
                    with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
                    json_cams = []
                    camlist = []
                    if scene_info.test_cameras:
                        camlist.extend(scene_info.test_cameras)
                    if scene_info.train_cameras:
                        camlist.extend(scene_info.train_cameras)
                    for id, cam in enumerate(camlist):
                        json_cams.append(camera_to_JSON(id, cam))
                    with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                        json.dump(json_cams, file)
                ```
                
        - 가우시안
            
            ```python
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                     "point_cloud",
                                                     "iteration_" + str(self.loaded_iter),
                                                     "point_cloud.ply"), args.train_test_exp)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
            ```
            
            - render의 경우 ply를 불러와 가우시안을 생성
            - train의 경우 Colmap으로 생성된 sparse한 포인트 클라우드로 가우시안을 생성(`create_from_pcd` )
2. **cameras.py**
    - Camera 클래스를 정의
        
        ```python
        class Camera(nn.Module):
            def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                         image_name, uid,
                         trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                         train_test_exp = False, is_test_dataset = False, is_test_view = False
                         ):
                super(Camera, self).__init__()
        ```
        
    - **class variables**
        - *uid, colmap_id, R, T, FoVx, FoVy, image_name*
        - *alpha_mask*
        - *original_image, image_width, image_height*
        - *invdepthmap, depth_reliable*
        - *zfar, znear*
        - *trans, scale*
        - *world_view_transform (R, T, trans, scale)*
        - *projection_matrix (znear, zfar, FoVx, FoVy)*
        - *full_proj_transform (view&proj)*
        - *camera_center(=camera pose)*
3. **colmap_loader.py**
    - Colmap의 결과물(bin, txt)을 변환함
4. **dataset_readers.py**
    - CameraInfo, SceneInfo 클래스를 정의함
    - colmap의 카메라를 CameraInfo로 불러옴 (`readColmapCameras`)
    - colmap의 결과물을 SceneInfo로 불러옴  (`readColmapSceneInfo`)
5. **gaussian_model.py**
    - GaussianModel 클래스를 정의
        - `setup_functions` 함수
            - `build_covariance_from_scaling_rotation`
            - activation function을 정의
        - class variables
            - *SH degree (active/max)*
            - *optimizer / optimizer type*
            - *xyz(위치)*
            - *features_dc, features_rest (SH)*
            - *scaling*
            - *rotation*
            - *opacity*
            - *max_radii2D*
            - *xyz_gradient_accum*
            - *denom*
            - *percent_dense*
            - *spatial_lr_scale*
        - class functions
            - capture/restore
            - `create_from_pcd`
                - train시 sparse한 포인트클라우드에서 가우시안을 초기화
                
                ```python
                    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
                        self.spatial_lr_scale = spatial_lr_scale
                        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
                        ### RGB -> SH -> features
                        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
                        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
                        features[:, :3, 0 ] = fused_color
                        features[:, 3:, 1:] = 0.0
                
                        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
                				
                				### dist: point cloud에서 가져옴
                				### scale: average distance of 3NN, activation is exp so log is added
                				### rot: 0으로 초기화
                        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
                        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
                        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                        rots[:, 0] = 1
                				
                				### opacity: inverse sigmoid
                        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
                				
                				### create nn.Parameter
                        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
                        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
                        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
                        self._scaling = nn.Parameter(scales.requires_grad_(True))
                        self._rotation = nn.Parameter(rots.requires_grad_(True))
                        self._opacity = nn.Parameter(opacities.requires_grad_(True))
                        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
                        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
                        self.pretrained_exposures = None
                        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
                        self._exposure = nn.Parameter(exposure.requires_grad_(True))
                ```
                
            - `load_ply`
                - render시 ply를 불러와 가우시안으로 바꿈
            - `save_ply`
                - 가우시안을 ply로 변환하여 저장
            - `reset_opacity`
            - `densify_and_split`, `densify_and_clone`,  `densify_and_prune`
                - `prune_points`
                    - optimizer을 prune함
                    - 그 외 다른 변수들도 prune함 (xyz_gradient_accum, denom, max_radii2D, tmp_radii)
                - `densification_postfix`
                    - densify(clone, split)후 새로운 가우시안 optimizer을 추가함

## gaussian_renderer

1. __init__.py
    - `render` 함수: diff_gaussian_rasterization의 `GaussianRasterizer` 을 이용해 가우시안을 2D 이미지로 렌더함
    - output:
        
        ```python
        out = {
                "render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : (radii > 0).nonzero(),
                "radii": radii,
                "depth" : depth_image
                }
        return out
        ```
        
2. network_gui.py
    - 실시간 렌더러에 쓰이는 로컬 네트워크 구현 코드

## utils

1. camera_utils.py
    - 카메라를 불러오는 함수
    - CamInfo로부터 카메라 리스트를 만드는 함수
    - json으로 바꾸기 위해 dictionary를 정의하는 함수
2. general_utils.py
    - inverse sigmoid
    - pil to torch
    - exponential lr
    - rotation
3. graphics_utils.py
    - view matrix, projection matrix
4. image_utils.py
    - mse, psnr
5. loss_utils.py
    - loss에 관련된 함수
6. make_depth_scale.py
    - 큰 범위의 환경일 경우 스케일을 줄이는 함수
7. read_write_model.py
    - bin/txt에서 byte로 read/write
8. sh_utils.py
    - spherical harmonics
9. system_utils.py
    - mkdir
