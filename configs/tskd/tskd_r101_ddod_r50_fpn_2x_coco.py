
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

teacher_ckpt = 'teacher weight'

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32
    )

model = dict(
    type='TSKDDDOD',
    data_preprocessor=data_preprocessor,
    teacher_config='../configs/ddod/ddod_r101_fpn_1x_coco.py',
    teacher_ckpt=teacher_ckpt,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='DDODHead',
        num_classes=6,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_iou=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    kd_cfg=dict(
        loss_cls_kd=dict(type='KDQualityFocalLoss', beta=1, loss_weight=1.0),
        loss_reg_kd=dict(type='GIoULoss', loss_weight=1.0),
        loss_iou_kd=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_feat_kd=dict(type='TSKDLoss', loss_weight=0.75),
        reused_teacher_head_idx=3),

    train_cfg=dict(
        # assigner is mean cls_assigner
        assigner=dict(type='ATSSAssigner', topk=9, alpha=0.8),
        reg_assigner=dict(type='ATSSAssigner', topk=9, alpha=0.5),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),

    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
    )
data = dict(persistent_workers=True)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
    )
# This `persistent_workers` is only valid when PyTorch>=1.7.0


train_dataloader = dict(batch_size=4, num_workers=4)
auto_scale_lr = dict(enable=True, base_batch_size=16)