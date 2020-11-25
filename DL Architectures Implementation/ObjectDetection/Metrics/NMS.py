import torch
from IoU import intersection_over_union

def nms(bboxes, iou_threshold, threshold, box_format = "corners"): 
    '''
    Does Non-Max suppression given bboxes

    Parameters: 
        bboxes (list): List os lists containing all bboxes with each bboxes specified as
        [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold

    STEPS: 
        (1) Start with discarding all BOUNDING BOXES < Prob Threshold
        
        (2) While BoundingBoxes:
            - Take out the largest probability box (sort the bboxes list first with reverse = True)
            - REMOVE all other boxes with IoU > Threshold

        Do this for EACH CLASS
    '''
    
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold] # Discard all bboxes < prob_threshold
    bboxes_after_nms = []

    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True)
    
    while bboxes: 
        chosen_box = bboxes.pop(0)

        bboxes = [
            box 
            for box in bboxes 
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold                   # REMOVE all other boxes with IoU > Threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

##################################################
#                  IMP EXPLANATION
##################################################

'''
(1) Sort the predictions by the confidence scores.

(2) Start from the top scores, ignore any current prediction if we find any previous predictions 
    that have the same class and IoU > 0.5 with the current prediction.

(3) Repeat step 2 until all predictions are checked.
'''