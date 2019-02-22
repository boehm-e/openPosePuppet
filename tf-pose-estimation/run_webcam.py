import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from tf_pose import common
from tf_pose.common import CocoPart

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)


        image_h, image_w = image.shape[:2]
        if len(humans) == 1:
            human = humans[0]
            # draw point
            print(human.body_parts.keys())
            if  1 in human.body_parts.keys() \
                and 2 in human.body_parts.keys() \
                and 3 in human.body_parts.keys() \
                and 5 in human.body_parts.keys() \
                and 6 in human.body_parts.keys():


                # 1
                body_part1 = human.body_parts[1]
                center1 = (int(body_part1.x * image_w + 0.5), int(body_part1.y * image_h + 0.5))

                # 2
                body_part2 = human.body_parts[2]
                center2 = (int(body_part2.x * image_w + 0.5), int(body_part2.y * image_h + 0.5))

                # 3
                body_part3 = human.body_parts[3]
                center3 = (int(body_part3.x * image_w + 0.5), int(body_part3.y * image_h + 0.5))

                # 5
                body_part5 = human.body_parts[5]
                center5 = (int(body_part5.x * image_w + 0.5), int(body_part5.y * image_h + 0.5))

                # 6
                body_part6 = human.body_parts[6]
                center6 = (int(body_part6.x * image_w + 0.5), int(body_part6.y * image_h + 0.5))

                import numpy as np
                def get_angle(p0, p1=np.array([0,0]), p2=None):
                    ''' compute angle (in degrees) for p0p1p2 corner
                    Inputs:
                        p0,p1,p2 - points in the form of [x,y]
                    '''
                    if p2 is None:
                        p2 = p1 + np.array([1, 0])
                    v0 = np.array(p0) - np.array(p1)
                    v1 = np.array(p2) - np.array(p1)

                    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
                    return np.degrees(angle)
                print("ANGLE 1 ==> ", get_angle(center1, center2, center3))
                print("ANGLE 2 ==> ", -get_angle(center1, center5, center6))


                angleLeft = get_angle(center1, center2, center3)
                angleRight = get_angle(center1, center5, center6)
                print("ANGLE 1 ==> ", angleLeft)
                print("ANGLE 2 ==> ", angleRight)

                f = open('/tmp/coords.txt', 'w')
                f.truncate(0)
                f.write(str(angleLeft)+ '\n')
                f.write(str(-angleRight))
                f.close()

                # print(f"CENTER[{1}] {center}")


        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
