import numpy as np


def lastNonZero(arr, axis, invalid_val=-1):
    # arr=arr[0]
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)
    xAxis = np.tile(np.arange(arr.shape[axis - 1], dtype=np.int64), (arr.shape[0], 1))
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def calEndTimeLB(temp1, dur_cp):
    x, y = lastNonZero(temp1, -1, invalid_val=-1)
    dur_cp[np.where(temp1 != 0)] = 0
    ids = np.arange(temp1.shape[0]).repeat(x.shape[0] // temp1.shape[0])
    dur_cp[ids, x, y] = temp1[ids, x, y]
    temp2 = np.cumsum(dur_cp, axis=-1)
    temp2[np.where(temp1 != 0)] = 0
    ret = temp1 + temp2
    return ret


if __name__ == "__main__":
    dur = np.array([[1, 2], [3, 4]])
    temp1 = np.zeros_like(dur)

    temp1[0, 0] = 1
    temp1[1, 0] = 3
    temp1[1, 1] = 5
    print(temp1)

    ret = calEndTimeLB(temp1, dur)
