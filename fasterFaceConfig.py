from frcnn.config import Config
from frcnn import utils
import frcnn.model as modellib
from frcnn import visualize
from frcnn.model import log

class FaceConfig(Config):
    """Configuration for training Face detection on the dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = "Face"
    
    # Train on 1 GPU and 1 image per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 1 + 1  # background + 1 Face classes

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.9
    STEPS_PER_EPOCH = 100
