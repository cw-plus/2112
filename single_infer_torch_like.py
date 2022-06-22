import os
import cv2
import time
import tqdm
import argparse
import numpy as np
import tensorflow as tf

from cfg.db_config import cfg
from shapely.geometry import Polygon
from lib.postprocess.post_process import SegDetectorRepresenter
import lib.networks.model_like_torch as model


def get_args():
    parser = argparse.ArgumentParser(description='DB-tf')
    # parser.add_argument('--ckpt_path', default='ckpt/torch_pretrain/DB_mv3_large_prune_asppmy_model.ckpt-0',
    #                     type=str,
    #                     help='load model')

    parser.add_argument('--ckpt_path', default='ckpt/DB_mv3_large_prune_aspp_model.ckpt-0',
                        type=str,
                        help='load model')
    parser.add_argument('--img_path', default='./figures/quant_issue1.jpg',
                        type=str)
    parser.add_argument('--gpu_id', default='0',
                        type=str)
    parser.add_argument('--is_poly', default=False,
                        type=bool)
    parser.add_argument('--show_res', default=True,
                        type=bool)

    args = parser.parse_args()

    return args


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class DB:

    def __init__(self, ckpt_path, gpu_id='0'):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        tf.reset_default_graph()
        self._input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')

        self._binarize_map, self._threshold_map, self._thresh_binary = model.model(self._input_images,
                                                                                   is_training=False)

        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)
        saver.restore(self.sess, ckpt_path)
        self.decoder = SegDetectorRepresenter()
        print('restore model from:', ckpt_path)

    def __del__(self):
        self.sess.close()

    def detect_img(self, img_path, is_poly=True, show_res=True):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        resized_img, ratio, size = self._resize_img(img)
        print("ratio: ", ratio, size)

        RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

        # reverse = np.array(resized_img * 255. + RGB_MEAN).astype(np.uint8)
        # cv2.imshow('sssssss', reverse)
        # cv2.waitKey(0)

        s = time.time()
        binarize_map, threshold_map, thresh_binary = self.sess.run(
            [self._binarize_map, self._threshold_map, self._thresh_binary],
            feed_dict={self._input_images: [resized_img]})
        net_time = time.time() - s

        s = time.time()
        boxes, scores = self.decoder([resized_img], binarize_map, is_poly)
        print("boxes: ", boxes)
        boxes = boxes[0]
        area = h * w
        res_boxes = []
        res_scores = []
        for i, box in enumerate(boxes):
            box[:, 0] *= ratio[1]
            box[:, 1] *= ratio[0]

            print("cfg.FILTER_MIN_AREA * area: ", cfg.FILTER_MIN_AREA * area)
            if Polygon(box).convex_hull.area > cfg.FILTER_MIN_AREA * area:
                res_boxes.append(box)
                res_scores.append(scores[0][i])
        post_time = time.time() - s

        if show_res:
            img_name = os.path.splitext(os.path.split(img_path)[-1])[0]
            make_dir('./show')
            cv2.imwrite('show/' + img_name + '_binarize_map.jpg', binarize_map[0][0:size[0], 0:size[1], :] * 255)
            cv2.imwrite('show/' + img_name + '_threshold_map.jpg', threshold_map[0][0:size[0], 0:size[1], :] * 255)
            cv2.imwrite('show/' + img_name + '_thresh_binary.jpg', thresh_binary[0][0:size[0], 0:size[1], :] * 255)

            print("res_boxes: ", res_boxes)
            img_ = img.copy().astype(np.uint8)
            for box in res_boxes:
                cv2.polylines(img_, [box.astype(np.int).reshape([-1, 1, 2])], True, (0, 255, 0))
                # print(Polygon(box).convex_hull.area, Polygon(box).convex_hull.area/area)
            cv2.imwrite('show/' + img_name + '_show.jpg', img_)

        return res_boxes, res_scores, (net_time, post_time)

    def _resize_img(self, img, max_size=1280):
        h, w, _ = img.shape

        ratio = float(max(h, w)) / max_size

        new_h = int((h / ratio // 64) * 64)
        new_w = int((w / ratio // 64) * 64)

        resized_img = np.array(cv2.resize(img, dsize=(new_w, new_h)), np.float32)

        RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        resized_img -= RGB_MEAN
        resized_img /= 255.

        ratio_w = w / new_w
        ratio_h = h / new_h

        return resized_img, (ratio_h, ratio_w), (new_h, new_w)


if __name__ == "__main__":
    args = get_args()

    db = DB(args.ckpt_path, args.gpu_id)

    db.detect_img(args.img_path, args.is_poly, args.show_res)
