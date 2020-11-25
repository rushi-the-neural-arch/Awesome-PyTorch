import torch
import torch.nn as nn  

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):  
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S  
        self.B = B
        self.C = C

        self.lambda_coord = 5  
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):  
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox

        # Index 0-19 are class PROBABILITIES, index 20 is the Class SCORE, 21-25 are the 4 Bounding Box values (not including 25)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # Target stays the same!  

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0) # dim = 0 ---> along the COLUMNS
        iou_maxes, best_box = torch.max(ious, dim=0) 
        
        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        exists_box = target[..., 20].unsqueeze(3) # I_Obj_i ----> IDENTITY function as mentioned in Paper ---> returns 0,1   

        
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
 
        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.

        # exists_box to calculate only if theres actually any object in that grid/cell
        box_predictions = exists_box * (              
            best_box * predictions[..., 26:30] # If the second BBox is best then index == 1 so this line will work (similar to Log Reg loss fn)
            + (1 - best_box) * predictions[..., 21:25]  # If the first BBox is best then index == 0 so this line will work
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)))  # 1e-6 for numerical stability, 
        # Derivative of 0 = Infinity

        # (N, S, S, 25)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) --> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2)
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU

        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # (N, S, S, 1) ---> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim = 1), 
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        # For the SECOND index BBox
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # (N, S, S, 20) --> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., 20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper -----> Confidence Loss
            + self.lambda_noobj * no_object_loss  # forth row -----> Confidence Loss
            + class_loss  # fifth row ---> Classification Loss
        )

        return loss