import numpy as np
import cv2
from google.colab.patches import cv2_imshow

from detectron2.config import get_cfg

def plot_prediction(img, predictor):
    outputs = predictor(img)
    #Blue color in RGB
    color = (255, 0, 0) #Blue
    #line thickness of 2 px
    thickness = 2
    
    for x1, x2, y1, y2 in outputs["instances"].get_fields()["pred_boxes"].tensor.cpu().numpy():
        #Start coordinate
        #represent the top left corner of rectangle
        start_point = int(x1), int(y1)
        
        #Ending coordinate
        #represent the bottom right corner of rectangle
        end_point = int(x2), int(y2)
        
        #Using cv2.rectangle() method
        #Draw a rectangle with blue line borders of thickness of 2 px
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    #Display the output
    print("Table detection:")
    cv2_imshow(img)
    
    
def detectron_predictor():
    # Load the model
    cfg = get_cfg()
    #set yaml
    cfg.merge_from_file('/content/All_X152.yaml')
    #set model weights
    cfg.MODEL.WEIGHTS = '/content/model_final.pth' # Set path model .pth
    predictor = DefaultPredictor(cfg) 
    return predictor  
  
    
if __name__ == "__main__":
    # Load the image
    img = cv2.imread(r"..\documents\Cut.jpg")
    
    #create detectron config
   
    # Plot the prediction
    plot_prediction(img, predictor)
    