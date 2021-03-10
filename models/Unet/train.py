import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
num_epochs = 100
num_workers = 2
image_height = 160 # 1280 originally 
image_weight = 240 # 1918 originally
pin_memory = True
load_model = True
train_img_dir = "/home/sd/lijitao/project/dp_models/datasets/Unet/train_images"
train_mask_dir = "/home/sd/lijitao/project/dp_models/datasets/Unet/train_masks/"
val_img_dir = "/home/sd/lijitao/project/dp_models/datasets/Unet/val_images/"
val_mask_dir = "/home/sd/lijitao/project/dp_models/datasets/Unet/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
        A.Resize(height=image_height, width = image_weight),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean = [0.0, 0.0, 0.0], 
            std = [1.0, 1.0, 1.0], 
            max_pixel_value = 255.0,
        ),
        ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_weight),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels = 3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(
        train_img_dir, 
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size, 
        train_transform,
        val_transforms,
        num_workers,
        pin_memory,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        # check accuracy
        check_accuracy(val_loader, model, device=device)
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=device
        )


if __name__ == "__main__":
    main()



# import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm
# import torch.nn as nn
# import torch.optim as optim
# from model import UNET
# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_imgs,
# )

# # Hyperparameters etc.
# LEARNING_RATE = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16
# NUM_EPOCHS = 3
# NUM_WORKERS = 2
# IMAGE_HEIGHT = 160  # 1280 originally
# IMAGE_WIDTH = 240  # 1918 originally
# PIN_MEMORY = True
# LOAD_MODEL = False
# TRAIN_IMG_DIR = "/home/sd/lijitao/project/dp_models/datasets/Unet/train_images/"
# TRAIN_MASK_DIR = "/home/sd/lijitao/project/dp_models/datasets/Unet/train_masks/"
# VAL_IMG_DIR = "/home/sd/lijitao/project/dp_models/datasets/Unet/val_images/"
# VAL_MASK_DIR = "/home/sd/lijitao/project/dp_models/datasets/Unet/val_masks/"

# def train_fn(loader, model, optimizer, loss_fn, scaler):
#     loop = tqdm(loader)

#     for batch_idx, (data, targets) in enumerate(loop):
#         data = data.to(device=DEVICE)
#         targets = targets.float().unsqueeze(1).to(device=DEVICE)

#         # forward
#         with torch.cuda.amp.autocast():
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)

#         # backward
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # update tqdm loop
#         loop.set_postfix(loss=loss.item())


# def main():
#     train_transform = A.Compose(
#         [
#             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#             A.Rotate(limit=35, p=1.0),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.1),
#             A.Normalize(
#                 mean=[0.0, 0.0, 0.0],
#                 std=[1.0, 1.0, 1.0],
#                 max_pixel_value=255.0,
#             ),
#             ToTensorV2(),
#         ],
#     )

#     val_transforms = A.Compose(
#         [
#             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#             A.Normalize(
#                 mean=[0.0, 0.0, 0.0],
#                 std=[1.0, 1.0, 1.0],
#                 max_pixel_value=255.0,
#             ),
#             ToTensorV2(),
#         ],
#     )

#     model = UNET(in_channels=3, out_channels=1).to(DEVICE)
#     loss_fn = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     train_loader, val_loader = get_loaders(
#         TRAIN_IMG_DIR,
#         TRAIN_MASK_DIR,
#         VAL_IMG_DIR,
#         VAL_MASK_DIR,
#         BATCH_SIZE,
#         train_transform,
#         val_transforms,
#         NUM_WORKERS,
#         PIN_MEMORY,
#     )

#     if LOAD_MODEL:
#         load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


#     check_accuracy(val_loader, model, device=DEVICE)
#     scaler = torch.cuda.amp.GradScaler()

#     for epoch in range(NUM_EPOCHS):
#         train_fn(train_loader, model, optimizer, loss_fn, scaler)

#         # save model
#         checkpoint = {
#             "state_dict": model.state_dict(),
#             "optimizer":optimizer.state_dict(),
#         }
#         save_checkpoint(checkpoint)

#         # check accuracy
#         check_accuracy(val_loader, model, device=DEVICE)

#         # print some examples to a folder
#         save_predictions_as_imgs(
#             val_loader, model, folder="saved_images/", device=DEVICE
#         )


# if __name__ == "__main__":
#     main()
