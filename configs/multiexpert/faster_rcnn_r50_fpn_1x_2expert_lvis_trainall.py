# 图片级别的类别统计
cls_num_list = [0, 0, 1, 21, 87, 3, 8, 0, 3, 0, 0, 1, 15, 46, 0, 0, 0, 24, 4, 2, 37, 0, 1, 1, 91, 6, 3, 2, 6, 3, 34, 1, 11, 2, 93, 90, 85, 4, 0, 0, 0, 16, 0, 8, 2, 92, 0, 2, 5, 0, 31, 0, 0,
                4, 2, 2, 18, 26, 86, 70, 72, 60, 4, 1, 0, 0, 14, 22, 3, 45, 1, 1, 0, 2, 0, 0, 3, 10, 39, 68, 1, 68, 3, 0, 18, 4, 0, 5, 7, 86, 1, 93, 0, 2, 1, 90, 18, 5, 0, 86, 2, 0, 5, 0, 21, 
                0, 0, 0, 1, 8, 93, 2, 18, 0, 0, 4, 0, 1, 85, 0, 0, 1, 0, 1, 6, 0, 93, 1, 4, 4, 0, 1, 44, 75, 0, 2, 1, 20, 9, 92, 0, 1, 1, 0, 0, 5, 64, 0, 3, 1, 0, 8, 5, 3, 57, 1, 9, 3, 1,
                0, 32, 4, 6, 0, 0, 8, 0, 0, 0, 0, 21, 1, 8, 1, 92, 0, 0, 11, 3, 58, 9, 0, 1, 91, 0, 39, 7, 15, 6, 2, 0, 70, 2, 2, 14, 0, 0, 54, 11, 0, 1, 5, 1, 0, 4, 2, 0, 2, 16, 1, 3, 93,
                4, 0, 0, 2, 4, 0, 1, 0, 3, 58, 4, 5, 3, 1, 0, 0, 0, 93, 3, 0, 1, 1, 4, 93, 0, 93, 0, 0, 10, 1, 0, 1, 5, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 24, 31, 12, 18, 4, 0, 0,
                13, 2, 6, 0, 0, 3, 1, 1, 4, 92, 51, 1, 2, 0, 10, 31, 4, 2, 1, 2, 0, 9, 39, 0, 0, 3, 1, 1, 0, 0, 3, 1, 92, 1, 42, 43, 0, 0, 5, 1, 0, 25, 6, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                1, 10, 1, 6, 1, 11, 3, 0, 0, 1, 1, 2, 1, 0, 2, 2, 0, 1, 2, 0, 0, 1, 0, 13, 0, 91, 1, 11, 1, 0, 89, 13, 0, 1, 0, 0, 1, 0, 0, 0, 13, 4, 0, 67, 1, 1, 0, 1, 0, 39, 1, 3, 1, 0, 8, 
                21, 0, 0, 5, 4, 91, 14, 9, 0, 1, 3, 2, 0, 76, 6, 39, 0, 0, 71, 1, 74, 3, 6, 9, 0, 0, 0, 0, 2, 0, 5, 0, 0, 10, 1, 5, 1, 0, 1, 29, 0, 68, 2, 1, 0, 7, 3, 2, 4, 0, 0, 80, 87, 1, 5, 
                1, 0, 0, 0, 34, 92, 0, 0, 4, 1, 1, 5, 5, 1, 0, 6, 4, 7, 2, 21, 69, 5, 1, 1, 0, 2, 91, 13, 2, 0, 1, 3, 0, 10, 0, 88, 1, 0, 2, 0, 2, 0, 1, 93, 1, 1, 0, 1, 77, 1, 1, 0, 7,
                0, 1, 0, 0, 2, 2, 5, 0, 1, 5, 0, 0, 3, 0, 1, 3, 1, 92, 1, 92, 1, 93, 7, 59, 0, 0, 1, 0, 2, 0, 0, 0, 15, 0, 2, 3, 0, 9, 2, 0, 0, 0, 0, 3, 3, 0, 8, 5, 0, 0, 7, 0, 1, 10, 1, 2,
                4, 0, 0, 5, 0, 12, 6, 0, 2, 92, 1, 0, 0, 92, 0, 0, 5, 37, 29, 92, 3, 3, 0, 0, 1, 5, 3, 92, 1, 2, 9, 1, 2, 2, 9, 1, 8, 3, 92, 5, 0, 0, 6, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 1, 5, 2, 
                0, 8, 2, 1, 91, 0, 93, 2, 0, 92, 0, 1, 0, 1, 1, 1, 0, 7, 13, 0, 0, 0, 45, 4, 77, 3, 3, 21, 93, 0, 0, 35, 0, 0, 1, 22, 5, 1, 3, 0, 91, 35, 22, 2, 10, 93, 0, 3, 0, 0, 0, 5, 13, 0, 
                24, 90, 6, 5, 50, 0, 8, 0, 0, 0, 1, 1, 1, 1, 0, 23, 1, 7, 57, 2, 0, 23, 19, 0, 8, 1, 0, 3, 2, 13, 6, 6, 1, 0, 2, 0, 31, 11, 1, 1, 10, 4, 1, 0, 4, 1, 19, 0, 41, 0, 3, 27, 1, 93, 3, 
                4, 2, 66, 1, 3, 18, 0, 0, 93, 10, 1, 75, 14, 7, 39, 15, 0, 0, 0, 1, 91, 0, 90, 93, 0, 0, 0, 1, 0, 1, 17, 6, 0, 0, 3, 0, 0, 1, 1, 1, 15, 47, 3, 0, 1, 8, 2, 1, 1, 0, 2, 4, 3, 0, 1, 
                51, 2, 3, 19, 0, 3, 3, 3, 0, 15, 19, 1, 0, 4, 1, 3, 0, 0, 27, 1, 0, 0, 0, 6, 0, 4, 3, 1, 7, 1, 0, 2, 13, 6, 0, 0, 0, 0, 0, 0, 11, 6, 2, 0, 93, 0, 0, 0, 0, 1,
                4, 9, 25, 4, 6, 1, 93, 0, 13, 2, 0, 0, 1, 37, 0, 1, 14, 0, 93, 21, 92, 1, 1, 1, 2, 2, 0, 0, 0, 91, 1, 39, 1, 2, 1, 20, 0, 2, 0, 27, 30, 46, 18, 1, 2, 0, 0, 3, 16, 1, 4, 24,
                0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 5, 0, 0, 4, 0, 8, 8, 4, 0, 0, 2, 4, 1, 0, 0, 3, 14, 3, 3, 0, 0, 1, 67, 1, 0, 0, 48, 0, 0, 0, 3, 1, 0, 2, 0, 2, 2, 0, 37, 33, 4, 0, 0, 10, 17,
                0, 0, 0, 0, 1, 24, 36, 44, 1, 1, 6, 11, 1, 0, 5, 0, 30, 6, 28, 8, 2, 0, 1, 1, 9, 3, 0, 1, 0, 7, 0, 0, 1, 0, 7, 2, 0, 1, 0, 1, 0, 1, 43, 0, 0, 1, 93, 93, 9, 2, 93, 2, 4, 1, 18, 22, 
                0, 1, 86, 1, 71, 93, 0, 89, 52, 15, 93, 47, 1, 1, 0, 3, 0, 0, 36, 1, 0, 44, 19, 93, 0, 2, 93, 0, 2, 0, 12, 0, 4, 1, 0, 1, 0, 21, 0, 92, 5, 0, 3, 61, 0, 3, 0, 2, 0, 25,
                2, 0, 0, 9, 0, 2, 1, 4, 0, 0, 0, 1, 16, 0, 11, 68, 2, 11, 26, 18, 93, 81, 1, 0, 0, 2, 0, 4, 2, 91, 3, 0, 92, 2, 4, 0, 2, 51, 23, 0, 21, 3, 0, 0, 0, 93, 2, 67, 0, 0, 24, 58,
                1, 0, 2, 21, 7, 1, 0, 18, 0, 2, 0, 2, 2, 4, 80, 48, 1, 29, 0, 0, 93, 60, 93, 0, 0, 1, 3, 0, 2, 0, 2, 0, 1, 1, 13, 1, 7, 3, 14, 3, 92, 48, 43, 7, 2, 26, 9, 2, 1, 1, 1, 24, 18,
                37, 4, 93, 2, 8, 92, 1, 28, 0, 0, 0, 1, 7, 93, 49, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 93, 2, 0, 8, 0, 5, 1, 91, 1, 30, 1, 1, 0, 0, 1, 1, 4, 0, 1, 4, 1, 8, 91, 7, 0, 0, 0, 1, 92,
                40, 0, 3, 0, 0, 0, 0, 1, 1, 1, 3, 5, 5, 3, 8, 0, 56, 92, 2, 4, 0, 1, 0, 3, 1, 1, 4, 28, 1, 22, 3, 39, 0, 2, 2, 0, 3, 3, 1, 4, 32, 0, 1, 3, 1, 85, 1]

# model settings
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),

    bbox_head=dict(
        type='MutiExpertBBoxHead',
        num_shared_convs=2,
        num_cls_convs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=1231,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='DiverseExpertLoss', cls_num_list=cls_num_list),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
)

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        score_thr=0.0,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=300)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'LvisDataset'
data_root = 'data/lvis/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'lvis_v0.5_train.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'lvis_v0.5_val.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'lvis_v0.5_val.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
        
# optimizer
# frozen_layers = ['backbone', 'neck', 'rpn_head']
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/baselines/faster_rcnn_r50_fpn_1x_lr2e2_lvis_2expert_trainall'
# load_from = './data/pretrained_models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
load_from = './data/pretrained_models/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
resume_from = None
evaluation = dict(interval=1, metrix='bbox')
workflow = [('train', 1)]
