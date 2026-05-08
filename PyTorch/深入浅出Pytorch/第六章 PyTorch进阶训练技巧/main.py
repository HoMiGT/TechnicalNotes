# import timm
# import torch
#
# avail_pretrained_models = timm.list_models(pretrained=True)
# # print(avail_pretrained_models)
#
# all_densenet_models = timm.list_models("*densenet*")
# # print(all_densenet_models)
#
# model = timm.create_model('resnet34', num_classes=10, pretrained=True)
# print(list(dict(model.named_children())['conv1'].parameters()))
#
# model = timm.create_model('resnet34',num_classes=10,pretrained=True)
# x = torch.randn(1,3,224,224)
# output = model(x)
# print(output)
# print(output.shape)
#
# # x = torch.randn(1,1,224,224)
# # output = model(x)
# # print(output)
#
# torch.save(model.state_dict(), "./timm_model.pth")
# model.load_state_dict(torch.load("./timm_model.pth"))
# print(list(dict(model.named_children())['conv1'].parameters()))

import albumentations as A
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = np.random.randint(0, 256, (100,100,3), dtype=np.uint8)
# transform = A.HorizontalFlip(p=1.0)
# transformed_data = transform(image=image)
# transformed_image = transformed_data["image"]
#
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(image)  # RGB
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.title("Flipped")
# plt.imshow(transformed_image)  # RGB
# plt.axis("off")
#
# plt.tight_layout()
# plt.show()
#
# print(f"Original shape: {image.shape}, Transformed shape: {transformed_image.shape}")
#
# result = transform(image=image)
# transformed_image = result["image"]

import numpy as np

bboxes = np.array([
    [23, 74, 295, 388, 1, 17],   # coords + class_id + track_id
    [377, 294, 252, 161, 2, 23],
], dtype=np.float32)

bbox_params = A.BboxParams(format='coco')


