import numpy as np


def crop(img: np.ndarray, newsize: np.ndarray):
    diff_rows = img.shape[0] - newsize[0]
    diff_cols = img.shape[0] - newsize[0]
    crop_rows = diff_rows // 2
    crop_cols = diff_cols // 2
    return img[crop_rows:-crop_rows, crop_cols:-crop_cols]



# desired_field_size = 64
    # phases, outputs, outputs2, energ = load_experimental_dataset(file_path, length, randomized, add_fourier=True)
    # outputs = outputs.astype(np.float64)
    # pool_amount = int(outputs.shape[-2] // desired_field_size)
    # crop_amount = int((outputs.shape[-2] - (pool_amount * desired_field_size)) // 2)
    # if crop_amount > 0:
    #     outputs = outputs[:, crop_amount:-crop_amount, crop_amount:-crop_amount]
    # outputs = pooling_2d(outputs, kernel=(pool_amount, pool_amount), func=np.mean)
    # outputs = np.sqrt(outputs)