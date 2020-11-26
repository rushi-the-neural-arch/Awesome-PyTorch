import os
import pandas as pd  

annotations = pd.read_csv('/Users/rushirajsinhparmar/Downloads/archive/test.csv')
label_dir =  '/Users/rushirajsinhparmar/Downloads/archive/labels'
img_dir = '/Users/rushirajsinhparmar/Downloads/archive/images/'

for index in range(2):
    print("INDEX:", index)
    label_path = os.path.join(label_dir, annotations.iloc[index, 1])
    with open(label_path) as f:  
        for label in f.readlines():
            class_label, x, y, width, height = [
                float(x) if float(x) != int(float(x)) else int(x) 
                for x in label.replace("\n", "").split()
                
            ]
            print("X: ", x) 
            print(class_label, x, y, width, height)

    img_path = os.path.join(img_dir, annotations.iloc[index, 0])
    print(label_path, img_path)
