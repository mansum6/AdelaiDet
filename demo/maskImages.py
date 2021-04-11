from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from adet.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer

from matplotlib.image import imread
import scipy.misc
from PIL import Image  
import numpy as np
import argparse
import os
import tqdm
import torch

import multiprocessing as mp


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
def cropper(org_image_path, mask_array, out_file_name):
    num_instances = mask_array.shape[0]
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_array_instance = []
    img = imread(str(org_image_path))
    output = np.zeros_like(img)
    for i in range(num_instances):
        mask_array_instance.append(mask_array[:, :, i:(i+1)])
        #output = np.where(mask_array_instance[i] == False, 0, (np.where(mask_array_instance[i] == True, 255, img)))
        output=np.where(mask_array_instance[i] == True, 255,output)
    print(output[:,:,0].shape)
    print(img.shape)
    #im=Image.fromarray(np.where((output == 255, 0,img)))
    im = Image.fromarray(output[:,:,0])
    
    if im.mode != 'RGBA':
      im = im.convert('RGBA')
    img = Image.open(org_image_path)
    im = Image.composite(img, im, im) 
    return im

    
if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()


    cfg = setup_cfg(args)

    #demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            im = read_image(path, format="BGR")
            #*******  
            
            # Inference with a keypoint detection model
            predictor = DefaultPredictor(cfg)

            outputs = predictor(im)
            preds = outputs["instances"].pred_classes.to("cpu").tolist()
            # this will get the names of our classes in dataset..
            labels_ = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

            # Wanting to only extract chair and person
            retain_ = []
            # retain_.append(labels_.index("chair"))
            retain_.append(labels_.index("person"))

            # retaining only retain_ from preds
            my_masks = [x for x in preds if x in retain_]
            my_masks = torch.tensor(my_masks)
            outputs["instances"].pred_classes = my_masks 

            #print(outputs["instances"].pred_masks.to("cpu").numpy())
            v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            #cv2_imshow(out.get_image()[:, :, ::-1])





            mask_array = outputs["instances"].pred_masks.to("cpu").numpy()
            #print(outputs["instances"].pred_keypoints.to("cpu").numpy().shape)
            #print(mask_array.shape)

            #print(mask_array)
            #cv2.imwrite('mask.png', mask_array)
            #cropper('1.png', mask_array)
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                cropper(path, mask_array).save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
