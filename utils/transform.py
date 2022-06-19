import albumentations as A
# from torchvision import transforms

def get_transforms(train_mean_std=((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                    test_mean_std = ((0.49421428, 0.48513139, 0.45040909), (0.24665252, 0.24289226, 0.26159238))):
    train_transforms = A.Compose(
        [
            A.Normalize(train_mean_std[0], train_mean_std[1]),
            A.CropAndPad(px = 4, keep_size=False),
            A.RandomCrop(width=32, height=32),
            A.CoarseDropout(1, 16, 16, 1, 16, 16,fill_value=0.473363),
            A.Rotate(5)
        ])

    val_transforms = A.Compose(
        [
            A.Normalize(mean=test_mean_std[0], std=test_mean_std[1]),
        ])
    return train_transforms, val_transforms
