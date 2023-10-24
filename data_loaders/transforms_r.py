from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
)
def custom_select_fn(x):
    return x > 0

def get_transform(split):
    if split == "train":
        transform = Compose([
        LoadImaged(keys=["image","P_mask", "label"]),
        EnsureChannelFirstd(keys=["image","P_mask", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),  
        CropForegroundd(keys=["image", "P_mask","label"], source_key="label", select_fn=custom_select_fn, margin=0),
        Orientationd(keys=["image", "P_mask","label"], axcodes="RAS"),
        RandCropByPosNegLabeld(
            keys=["image", "P_mask","label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
            allow_smaller =True,),        
        ResizeWithPadOrCropd(keys=["image", "P_mask","label"], spatial_size=(96, 96, 96), mode=('constant')),
        ])
    elif split == "valid":
        transform = Compose([
        LoadImaged(keys=["image","P_mask", "label"]),
        EnsureChannelFirstd(keys=["image", "P_mask","label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image","P_mask", "label"], source_key="label", select_fn=custom_select_fn, margin=0),
        Orientationd(keys=["image", "P_mask","label"], axcodes="RAS"),
        ])
    elif split == "post":
        transform = Compose([
        LoadImaged(keys=["image","P_mask", "label"]),
        EnsureChannelFirstd(keys=["image", "P_mask","label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image","P_mask", "label"], source_key="label", select_fn=custom_select_fn, margin=0),
        Orientationd(keys=["image", "P_mask","label"], axcodes="RAS"),
        ]) 
    else:
        raise ValueError(f"split {split} is not supported")

    return transform