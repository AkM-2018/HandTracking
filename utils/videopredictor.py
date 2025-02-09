import os
import numpy as np
import torch
import supervision as sv
import cv2
import shutil
from sam2.build_sam import build_sam2_video_predictor

from .imageutil import ImageUtil

class VideoPredictor:
    """Initialize variables

    """
    def __init__(self):
        self.video_dir_path = './vids'
        colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.ColorPalette.from_hex(colors),
            color_lookup=sv.ColorLookup.TRACK)
        self.image_util = ImageUtil()
    
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
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        self.inference_state = self.predictor.init_state(video_path=self.video_dir_path)
        self.predictor.reset_state(self.inference_state)

    """Adds mask to the image
    """
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    """Shows the segmented image

    """
    def show_segmented_image(self, mask_logits, object_ids):
        frames_paths = sorted(sv.list_files_with_extensions(
            directory=self.video_dir_path, 
            extensions=["jpg"]))

        frame = cv2.imread(frames_paths[0])
        masks = (mask_logits > 0.0).cpu().numpy()
        N, X, H, W = masks.shape
        masks = masks.reshape(N * X, H, W)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            tracker_id=np.array(object_ids)
        )
        frame = self.mask_annotator.annotate(frame, detections)
        self.image_util.show_image(frame)

    """Add hand landmark label points to the predictor

    """
    def add_label_points(self, hand_landmark, landmark_negative, check_segment: bool):
        ann_frame_idx = 0
        ann_obj_id = 1
        hand_landmark_label = np.ones(len(hand_landmark), dtype=int)

        if len(landmark_negative) != 0:
            hand_landmark = np.append(hand_landmark, landmark_negative, axis=0)
            landmark_negative_label = np.zeros(len(landmark_negative), dtype=int)
            hand_landmark_label = np.append(hand_landmark_label, landmark_negative_label)

        print("hand_landmark", hand_landmark)
        print("hand_landmark_label", hand_landmark_label)

        points = np.array([hand_landmark], dtype=np.float32)
        labels = np.array([hand_landmark_label], np.int32)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        if check_segment == True:
            self.show_segmented_image(out_mask_logits, out_obj_ids)

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
                frame = self.mask_annotator.annotate(frame, detections)
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
