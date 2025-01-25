import os
import numpy as np
import torch
import supervision as sv
import cv2
import shutil
from sam2.build_sam import build_sam2_video_predictor

class VideoPredictor:
    """Initialize variables

    """
    def __init__(self):
        self.video_dir_path = './vids'
    
    """Initialize the device type to use for processing

    """
    def init_device(self, use_cpu: bool = False):
        if use_cpu == True:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")
    
    """Modify device type setting for cuda and mps

    """
    def modify_torch_settings_for_device(self):
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
    
    """Initialize predictor using SAM2 module and interface_state

    """
    def init_video_predictor(self):
        sam2_checkpoint = "../../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        self.inference_state = self.predictor.init_state(video_path=self.video_dir_path)
        self.predictor.reset_state(self.inference_state)

    """Add hand landmark label points to the predictor

    """
    def add_label_points(self, hand_landmark):
        ann_frame_idx = 0
        ann_obj_id = 1
        points = np.array([hand_landmark], dtype=np.float32)
        labels = np.array(np.ones(len(hand_landmark), dtype=int), np.int32)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    """Convert video into image frames and save them in a directory

    Args:
        source_video_path: path of the source video
    """
    def save_jpg_dir_from_vid(self, source_video_path: str):
        self.source_video_path = source_video_path
        self.create_empty_tmp_dir()
        frames_generator = sv.get_video_frames_generator(source_video_path)
        sink = sv.ImageSink(
            target_dir_path=self.video_dir_path,
            image_name_pattern="{:05d}.jpg")
        with sink:
            for frame in frames_generator:
                sink.save_image(frame)
        print(f"saved frames for video in: {self.video_dir_path}")

    """Creates an empty directory for storing image frames

    """
    def create_empty_tmp_dir(self):
        self.cleanup()
        os.makedirs(self.video_dir_path)

    """Propogates the required masks throughout the video and saves the video

    Args:
        target_video_path: A list of numerical values.
    """
    def propogate_in_video(self, target_video_path: str):
        colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
        mask_annotator = sv.MaskAnnotator(
            color=sv.ColorPalette.from_hex(colors),
            color_lookup=sv.ColorLookup.TRACK)
        
        video_info = sv.VideoInfo.from_video_path(self.source_video_path)
        frames_paths = sorted(sv.list_files_with_extensions(
            directory=self.video_dir_path, 
            extensions=["jpg"]))
        
        with sv.VideoSink(target_video_path, video_info=video_info) as sink:
            for frame_idx, object_ids, mask_logits in self.predictor.propagate_in_video(self.inference_state):
                frame = cv2.imread(frames_paths[frame_idx])
                masks = (mask_logits > 0.0).cpu().numpy()
                N, X, H, W = masks.shape
                masks = masks.reshape(N * X, H, W)
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks),
                    mask=masks,
                    tracker_id=np.array(object_ids)
                )
                frame = mask_annotator.annotate(frame, detections)
                sink.write_frame(frame)
        print(f"saved new video at: {target_video_path}")
    
    """Deletes the temporary image frames folder

    """
    def cleanup(self):
        if os.path.exists(self.video_dir_path):
            try:
                shutil.rmtree(self.video_dir_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
