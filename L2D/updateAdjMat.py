import numpy as np

from Params import configs


def getActionNbghs(action, opIDsOnMchs):
    coordAction = np.where(opIDsOnMchs == action)
    precd = opIDsOnMchs[0][
        coordAction[1][0],
        coordAction[2][0] - 1 if coordAction[2][0].item() > 0 else coordAction[1][0],
    ].item()
    succdTemp = opIDsOnMchs[0][
        coordAction[1][0],
        (
            coordAction[2][0] + 1
            if coordAction[2][0].item() + 1 < opIDsOnMchs.shape[-1]
            else coordAction[2][0]
        ),
    ].item()
    succd = action if succdTemp < 0 else succdTemp
    # precedX = coordAction[0]
    # precedY = coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]
    # succdX = coordAction[0]
    # succdY = coordAction[1] + 1 if coordAction[1].item()+1 < opIDsOnMchs.shape[-1] else coordAction[1]
    return precd, succd


if __name__ == "__main__":
    opIDsOnMchs = np.array(
        [
            [7, 29, 33, 16, -6, -6],
            [6, 18, 28, 34, 2, -6],
            [26, 31, 14, 21, 11, 1],
            [30, 19, 27, 13, 10, -6],
            [25, 20, 9, 15, -6, -6],
            [24, 12, 8, 32, 0, -6],
        ]
    )

    action = 16
    precd, succd = getActionNbghs(action, opIDsOnMchs)
    print(precd, succd)
