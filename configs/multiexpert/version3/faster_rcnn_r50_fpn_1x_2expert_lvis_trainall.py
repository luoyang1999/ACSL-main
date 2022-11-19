# 图片级别的类别统计
cls_num_list = [0, 2, 34, 212, 1052, 95, 26, 10, 23, 13, 9, 20, 230, 591, 4, 2, 6, 269, 23, 28, 360, 7, 4, 19, 1077, 68, 45, 24, 65, 24, 715, 5, 191, 20, 1100, 1096, 
    974, 32, 2, 2, 1, 179, 2, 116, 32, 947, 8, 32, 96, 2, 651, 4, 1, 107, 36, 18, 208, 439, 997, 1104, 932, 963, 28, 24, 4, 3, 179, 245, 32, 500, 2, 34, 2, 25, 1, 14, 13, 
    250, 339, 1025, 80, 785, 65, 2, 204, 43, 1, 139, 191, 1105, 183, 1117, 11, 43, 1, 1075, 276, 61, 12, 1023, 9, 5, 16, 18, 159, 5, 4, 2, 15, 65, 1003, 29, 146, 1, 72, 61, 
    1, 6, 1017, 15, 8, 26, 1, 27, 682, 6, 1090, 4, 49, 53, 4, 2, 782, 1070, 11, 21, 4, 266, 130, 1102, 1, 19, 3, 7, 6, 44, 993, 4, 61, 17, 1, 56, 31, 16, 797, 4, 52, 13, 19, 
    2, 394, 5, 50, 8, 1, 40, 41, 3, 12, 2, 338, 11, 119, 8, 1047, 23, 12, 100, 36, 1057, 111, 2, 8, 1025, 5, 389, 42, 104, 52, 22, 15, 856, 57, 27, 251, 17, 7, 666, 109, 3, 
    13, 54, 32, 5, 17, 10, 1, 64, 509, 9, 50, 1107, 122, 7, 1, 11, 30, 8, 4, 4, 30, 694, 75, 22, 24, 22, 7, 5, 13, 1074, 86, 1, 12, 19, 62, 1086, 1, 1106, 4, 6, 172, 7, 2, 3, 
    49, 6, 3, 33, 78, 10, 1, 7, 2, 4, 25, 1, 14, 19, 2, 1, 329, 392, 99, 122, 46, 4, 2, 91, 21, 270, 1, 9, 29, 2, 53, 18, 1092, 554, 28, 12, 1, 127, 430, 38, 10, 11, 15, 6, 
    160, 354, 22, 1, 25, 7, 29, 2, 6, 24, 2, 1082, 4, 556, 397, 2, 4, 94, 6, 2, 160, 58, 4, 10, 79, 4, 33, 49, 1, 9, 1, 23, 3, 4, 8, 137, 7, 24, 5, 138, 8, 7, 7, 21, 21, 17, 
    132, 8, 9, 39, 17, 25, 27, 51, 8, 18, 10, 141, 13, 958, 18, 88, 1, 1, 1084, 565, 1, 14, 1, 9, 6, 1, 8, 2, 120, 24, 11, 574, 7, 37, 1, 7, 7, 201, 4, 56, 37, 12, 94, 213, 8, 
    1, 93, 62, 1122, 340, 93, 3, 4, 14, 1, 2, 997, 46, 574, 1, 4, 1116, 11, 784, 30, 63, 86, 14, 3, 1, 2, 16, 7, 110, 8, 16, 129, 4, 39, 5, 1, 17, 325, 3, 1107, 25, 3, 1, 128, 
    4, 21, 21, 18, 1, 863, 1078, 5, 48, 11, 1, 5, 2, 358, 1083, 5, 3, 21, 9, 3, 29, 115, 27, 3, 102, 73, 93, 20, 336, 773, 65, 9, 6, 2, 23, 1096, 196, 12, 11, 7, 24, 1, 179, 14, 
    1043, 14, 10, 38, 13, 16, 3, 20, 1070, 8, 8, 10, 24, 1006, 21, 14, 2, 77, 1, 9, 11, 2, 9, 14, 23, 20, 3, 33, 4, 7, 16, 1, 32, 31, 12, 1073, 12, 1077, 36, 1073, 55, 830, 2, 10, 12, 3, 35, 3, 6, 7, 115, 1, 33, 21, 10, 77, 58, 3, 1, 1, 2, 21, 15, 6, 104, 57, 14, 3, 80, 13, 12, 126, 33, 15, 5, 3, 6, 89, 6, 144, 56, 5, 50, 1097, 5, 1, 2, 1084, 3, 1, 31, 512, 417, 1066, 39, 7, 20, 1, 28, 46, 27, 1084, 3, 44, 808, 3, 2, 25, 239, 14, 132, 123, 1092, 104, 2, 5, 22, 1, 2, 6, 3, 94, 35, 1, 21, 1, 1, 4, 49, 4, 5, 87, 13, 8, 1098, 10, 1090, 24, 2, 1066, 27, 16, 7, 8, 21, 3, 3, 63, 110, 1, 9, 15, 324, 12, 940, 21, 47, 464, 1077, 1, 5, 907, 4, 5, 6, 347, 61, 7, 61, 2, 1076, 376, 592, 33, 266, 1073, 2, 95, 7, 4, 38, 11, 207, 1, 385, 1079, 112, 140, 526, 1, 65, 4, 4, 16, 11, 16, 3, 8, 1, 449, 13, 85, 607, 27, 2, 215, 192, 9, 61, 7, 1, 12, 33, 160, 79, 82, 1, 6, 28, 2, 503, 149, 18, 8, 111, 53, 25, 18, 33, 8, 161, 1, 587, 3, 68, 264, 5, 1036, 43, 56, 25, 730, 34, 173, 146, 7, 2, 1043, 138, 8, 815, 176, 43, 458, 237, 2, 10, 5, 10, 1106, 1, 1103, 1141, 8, 13, 9, 12, 2, 10, 128, 43, 27, 2, 47, 2, 1, 11, 22, 7, 365, 476, 62, 1, 26, 97, 42, 25, 11, 1, 29, 84, 62, 1, 19, 597, 62, 15, 171, 6, 47, 7, 6, 4, 199, 295, 1, 3, 13, 7, 21, 6, 3, 284, 27, 17, 1, 10, 105, 3, 46, 35, 27, 79, 8, 5, 21, 190, 86, 2, 2, 5, 26, 3, 5, 80, 
    52, 11, 1, 1119, 6, 5, 7, 3, 14, 65, 126, 263, 35, 50, 3, 1093, 7, 102, 16, 1, 1, 7, 680, 4, 10, 145, 4, 1083, 291, 1088, 24, 9, 3, 17, 7, 2, 11, 13, 1071, 1, 350, 3, 
    22, 6, 147, 1, 22, 13, 457, 302, 773, 158, 30, 19, 24, 5, 55, 116, 13, 39, 280, 1, 2, 1, 2, 4, 47, 2, 2, 9, 1, 7, 197, 38, 2, 23, 2, 96, 71, 33, 20, 1, 31, 21, 53, 3, 
    19, 22, 492, 37, 16, 9, 3, 285, 707, 12, 6, 8, 870, 2, 3, 18, 38, 7, 6, 32, 3, 20, 136, 13, 760, 315, 165, 26, 2, 100, 75, 2, 10, 9, 3, 8, 242, 612, 442, 2, 4, 132, 130, 3, 4, 115, 2, 427, 37, 389, 73, 21, 1, 1, 30, 87, 47, 9, 5, 2, 20, 1, 12, 8, 6, 73, 3, 2, 2, 9, 14, 7, 3, 520, 1, 1, 9, 1086, 1078, 69, 25, 1110, 2, 68, 49, 236, 250, 4, 3, 1064, 19, 1014, 1140, 12, 1112, 961, 216, 1145, 673, 24, 11, 1, 40, 7, 2, 628, 22, 14, 361, 239, 1104, 3, 2, 1091, 4, 17, 3, 100, 1, 18, 14, 1, 3, 7, 197, 4, 
    1121, 35, 6, 67, 682, 11, 34, 5, 41, 8, 424, 33, 1, 1, 356, 1, 3, 21, 40, 3, 6, 153, 23, 164, 591, 186, 670, 38, 1066, 289, 194, 1100, 1033, 1, 4, 2, 4, 1, 81, 33, 1114, 32, 7, 1128, 14, 17, 20, 40, 560, 316, 16, 415, 23, 4, 1, 2, 1045, 44, 880, 3, 1, 726, 1080, 1, 2, 67, 552, 81, 18, 12, 177, 11, 37, 9, 32, 24, 81, 848, 470, 34, 524, 1, 30, 1102, 787, 1143, 1, 22, 25, 92, 2, 44, 18, 17, 12, 23, 40, 157, 10, 155, 33, 144, 61, 1109, 620, 692, 98, 20, 433, 123, 85, 26, 17, 31, 331, 364, 638, 37, 1122, 8, 91, 1022, 6, 511, 1, 5, 1, 7, 64, 1097, 467, 1, 32, 1, 22, 7, 25, 4, 11, 5, 7, 1111, 56, 1, 91, 10, 24, 23, 1110, 39, 782, 12, 3, 7, 3, 10, 2, 21, 3, 38, 24, 15
    , 38, 1074, 65, 1, 1, 3, 35, 1097, 407, 24, 42, 2, 5, 8, 1, 13, 28, 30, 18, 54, 108, 24, 62, 21, 800, 1093, 39, 39, 4, 5, 7, 26, 16, 23, 41, 1051, 11, 354, 9, 538, 1, 
    69, 13, 1, 43, 46, 23, 67, 338, 8, 6, 33, 2, 984, 44]

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
        type='MutiExpertBBoxHead3',
        num_shared_convs=1,
        num_cls_convs=2,
        num_reg_convs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=1231,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        ratio_li=[1, 1, 1],
        loss_cls=dict(
            type='DiverseExpertLoss2', cls_num_list=cls_num_list),
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
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        sample_factor=0.9,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'lvis_v0.5_val.json',
            img_prefix=data_root + 'val2017/',
            test_mode = 'True',
            pipeline=test_pipeline))
    # test=dict(
    #         type=dataset_type,
    #         ann_file=data_root + 'lvis_v0.5_val.json',
    #         img_prefix=data_root + 'val2017/',
    #         pipeline=test_pipeline)
)

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
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/head3/faster_rcnn_r50_fpn_1x_lr2e2_lvis_3expert_trainall'
# load_from = './data/pretrained_models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
load_from = './data/download_models/R50-baseline.pth'
resume_from = None
evaluation = dict(interval=1, metrix='bbox')
workflow = [('train', 1)]
