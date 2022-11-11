from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from lvis.lvis import LVIS

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default='demo/demo.jpg')
    parser.add_argument('--config', help='Config file', default='configs/baselines/faster_rcnn_r50_fpn_1x_lvis.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='work_dirs/baselines/faster_rcnn_r50_fpn_1x_lr2e2_lvis/epoch_12.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)

    lvis = LVIS('/data/dataset/lvis/lvis_v0.5_train.json')
    CLASSES = tuple([item['name'] for item in lvis.dataset['categories']])
    # print(CLASSES)

    # show the results
    show_result_pyplot(args.img, result, class_names=CLASSES, score_thr=args.score_thr)


if __name__ == '__main__':
    main()