#!/usr/bin/env python
# coding: utf-8

# In[ ]:


dependencies = ['torch', 'numpy']

from ultralytics.yolo import YOLO

def yolov8(pretrained=False, **kwargs):
    """YOLOv8 model with pretrained weights"""
    model = YOLO(cfg='models/yolov8.cfg', ch=3, nc=80)
    if pretrained:
        state_dict = torch.load('yolov8.pt', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model

