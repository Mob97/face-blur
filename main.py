import anyconfig
import munch
import os
import glob
import cv2

from face_detection import FaceDetector
from face_blurring import anonymize_face_pixelate

image_dir = "/home/minhbq/duongnh"
output_dir = "/home/minhbq/duongnh_blurred"
exts = ('*.jpg', '*.png', '*.jpeg')
image_list = []

def main():
    cfg = anyconfig.load('settings.yaml')
    cfg = munch.munchify(cfg)  
    detector = FaceDetector(cfg.face_detector.model_path, -1)
    for ext in exts:
        image_list.extend(glob.glob(os.path.join(image_dir, ext)))
    for image_path in image_list:
        filename = image_path.split('/')[-1] #os should be linux
        image = cv2.imread(image_path)
        bboxes, facial_lanmarks = detector.detect(image[:, :, ::-1], 1.0)
        for i, bbox in enumerate(bboxes):
            conf_score = bbox[4]
            if conf_score < cfg.face_detector.confident_score_threshold:
                continue
            xmin, ymin, xmax, ymax = [int(val) for val in bbox[:4]]
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            face = image[ymin:ymax, xmin:xmax]
            image[ymin:ymax, xmin:xmax] = anonymize_face_pixelate(face)
        cv2.imwrite(os.path.join(output_dir, filename), image)

    
    

 
if __name__ == "__main__":
    main()
