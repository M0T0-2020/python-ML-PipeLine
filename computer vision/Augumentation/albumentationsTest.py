"""
参考
https://qiita.com/kurilab/items/b69e1be8d0224ae139ad
https://farml1.com/albu/
"""
# install pip install albumentations

import albumentations as A
import numpy as np


def compose_augmentation():
    transform = [
        # リサイズ
        A.Resize(224,224),
        # ぼかし
        A.Blur(blur_limit=15, p=1.0),
        # 明るさ、コントラスト
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=1.0),
        # 回転
        A.RandomRotate90(p=0.5),
        #Random Erasing
        A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0)
    ]
    return A.Compose(transform)

if __name__ == "__main__":
    trn = compose_augmentation()
    w, h, d = 64, 64, 3
    x = np.random.rand((h, w, d))
    x:np.ndarray = (255*x).astype(np.uint8)
    x = trn(image=x)["image"]
