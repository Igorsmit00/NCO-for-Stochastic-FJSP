import numpy as np

from Params import configs


def permissibleLeftShift(a, durMat, mchMat, mchsStartTimes, opIDsOnMchs):
    n_m = mchMat.max()
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(
        a, mchMat, durMat, mchsStartTimes, opIDsOnMchs
    )
    dur_a = np.take(durMat, a)
    dur_a = durMat[:, a // n_m, a % n_m]
    mch_a = np.take(mchMat, a) - 1
    startTimesForMchOfa = mchsStartTimes[:, mch_a]
    opsIDsForMchOfa = opIDsOnMchs[:, mch_a]
    flag = False

    # possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)
    possiblePos = np.where(np.expand_dims(jobRdyTime_a, 1) < startTimesForMchOfa)
    # print('possiblePos:', possiblePos)
    if len(possiblePos[0]) == 0:
        startTime_a = putInTheEnd(
            a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa
        )
    else:
        # idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa, n_m)
        # print('legalPos:', legalPos)
        # if len(legalPos[0]) == 0:
        startTime_a = putInTheEnd(
            a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa
        )
        # else:
        #     flag = True
        #     startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa)
    return startTime_a, flag


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')
    index = np.where(startTimesForMchOfa == -configs.high)[0][0]
    where_result = np.where(startTimesForMchOfa == -configs.high)
    unique_a = np.unique(where_result[0])
    index = (
        unique_a,
        np.array([where_result[1][where_result[0] == val].min() for val in unique_a]),
    )
    startTime_a = np.maximum(jobRdyTime_a, mchRdyTime_a)
    startTimesForMchOfa[index] = startTime_a
    opsIDsForMchOfa[index] = a
    return startTime_a


def calLegalPos(
    dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa, n_m
):
    n_rea = durMat.shape[0]
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos].reshape(n_rea, -1)
    # durOfPossiblePos = np.take(durMat, opsIDsForMchOfa[possiblePos])
    durOfPossiblePos = durMat[
        possiblePos[0], possiblePos[1] // n_m, possiblePos[1] % n_m
    ].reshape(n_rea, -1)
    n_rea = durMat.shape[0]
    unique_first = np.unique(possiblePos[0])
    index = (
        unique_first,
        np.array([possiblePos[1][possiblePos[0] == val].min() for val in unique_first]),
    )
    startTimeEarlst = np.maximum(
        jobRdyTime_a,
        startTimesForMchOfa[index[0], index[1] - 1]
        + durMat[
            index[0],
            opsIDsForMchOfa[0][index[1] - 1] // n_m,
            opsIDsForMchOfa[0][index[1] - 1] % n_m,
        ],
    )
    # endTimesForPossiblePos
    endTimesForPossiblePos = np.concatenate(
        (
            np.reshape(startTimeEarlst, (n_rea, -1)),
            np.reshape(startTimesOfPossiblePos + durOfPossiblePos, (n_rea, -1)),
        ),
        axis=1,
    )[:, :-1]
    # assert endTimesForPossiblePos.shape == (n_rea, 1)
    # endTimesForPossiblePos = endTimesForPossiblePos.reshape(-1)
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a.reshape(n_rea, -1) <= possibleGaps)  # [0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(
    a,
    idxLegalPos,
    legalPos,
    endTimesForPossiblePos,
    startTimesForMchOfa,
    opsIDsForMchOfa,
):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(
        startTimesForMchOfa, earlstPos, startTime_a, axis=-1
    )[:, :-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a, axis=-1)[:, :-1]
    return startTime_a


def calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs):
    n_m = mchMat.max()
    mch_a = np.take(mchMat, a) - 1
    # cal jobRdyTime_a
    jobPredecessor = a - 1 if a % mchMat.shape[1] != 0 else None
    if jobPredecessor is not None:
        # durJobPredecessor = np.take(durMat, jobPredecessor)
        durJobPredecessor = durMat[
            np.arange(durMat.shape[0]), jobPredecessor // n_m, jobPredecessor % n_m
        ]
        mchJobPredecessor = np.take(mchMat, jobPredecessor) - 1
        jobRdyTime_a = (
            mchsStartTimes[np.arange(durMat.shape[0]), mchJobPredecessor][
                np.where(
                    opIDsOnMchs[np.arange(durMat.shape[0]), mchJobPredecessor]
                    == jobPredecessor
                )
            ]
            + durJobPredecessor
        )
    else:
        jobRdyTime_a = np.zeros((durMat.shape[0],))
    # cal mchRdyTime_a
    mchPredecessor = (
        opIDsOnMchs[0, mch_a][np.where(opIDsOnMchs[0, mch_a] >= 0)][-1]
        if len(np.where(opIDsOnMchs[0, mch_a] >= 0)[0]) != 0
        else None
    )
    if mchPredecessor is not None:
        durMchPredecessor = durMat[
            np.arange(durMat.shape[0]), mchPredecessor // n_m, mchPredecessor % n_m
        ]
        # durMchPredecessor = np.take(durMat, mchPredecessor)
        where_results = np.where(
            mchsStartTimes[np.arange(mchsStartTimes.shape[0]), mch_a] >= 0
        )
        unique_first = np.unique(where_results[0], return_inverse=False)
        last_indices = np.searchsorted(where_results[0], unique_first, side="right") - 1
        last_where_results = (unique_first, where_results[1][last_indices])
        mchRdyTime_a = (
            mchsStartTimes[np.arange(mchsStartTimes.shape[0]), mch_a][
                last_where_results
            ]
            + durMchPredecessor
        )
    else:
        mchRdyTime_a = np.zeros((durMat.shape[0],))

    return jobRdyTime_a, mchRdyTime_a


if __name__ == "__main__":
    import time

    from JSSP_Env import SJSSP
    from uniform_instance_gen import uni_instance_gen

    n_j = 3
    n_m = 3
    low = 1
    high = 99
    SEED = 10
    np.random.seed(SEED)
    env = SJSSP(n_j=n_j, n_m=n_m)

    """arr = np.ones(3)
    idces = np.where(arr == -1)
    print(len(idces[0]))"""

    # rollout env random action
    t1 = time.time()
    data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
    print("Dur")
    print(data[0])
    print("Mach")
    print(data[-1])
    print()

    # start time of operations on machines
    mchsStartTimes = -configs.high * np.ones_like(data[0].transpose(), dtype=np.int32)
    # Ops ID on machines
    opIDsOnMchs = -n_j * np.ones_like(data[0].transpose(), dtype=np.int32)

    # random rollout to test
    # count = 0
    _, _, omega, mask = env.reset(data)
    rewards = []
    flags = []
    # ts = []
    while True:
        action = np.random.choice(omega[np.where(mask == 0)])
        print(action)
        mch_a = np.take(data[-1], action) - 1
        # print(mch_a)
        # print('action:', action)
        # t3 = time.time()
        adj, _, reward, done, omega, mask = env.step(action)
        # t4 = time.time()
        # ts.append(t4 - t3)
        # jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a=action, mchMat=data[-1], durMat=data[0], mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs)
        # print('mchRdyTime_a:', mchRdyTime_a)
        startTime_a, flag = permissibleLeftShift(
            a=action,
            durMat=data[0].astype(np.single),
            mchMat=data[-1],
            mchsStartTimes=mchsStartTimes,
            opIDsOnMchs=opIDsOnMchs,
        )
        flags.append(flag)
        # print('startTime_a:', startTime_a)
        # print('mchsStartTimes\n', mchsStartTimes)
        # print('NOOOOOOOOOOOOO' if not np.array_equal(env.mchsStartTimes, mchsStartTimes) else '\n')
        print("opIDsOnMchs\n", opIDsOnMchs)
        # print('LBs\n', env.LBs)
        rewards.append(reward)
        # print('ET after action:\n', env.LBs)
        print()
        if env.done():
            break
    t2 = time.time()
    print(t2 - t1)
    # print(sum(ts))
    # print(np.sum(opIDsOnMchs // n_m, axis=1))
    # print(np.where(mchsStartTimes == mchsStartTimes.max()))
    # print(opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())])
    print(
        mchsStartTimes.max()
        + np.take(
            data[0], opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())]
        )
    )
    # np.save('sol', opIDsOnMchs // n_m)
    # np.save('jobSequence', opIDsOnMchs)
    # np.save('testData', data)
    # print(mchsStartTimes)
    durAlongMchs = np.take(data[0], opIDsOnMchs)
    mchsEndTimes = mchsStartTimes + durAlongMchs
    print(mchsStartTimes)
    print(mchsEndTimes)
    print()
    print(env.opIDsOnMchs)
    print(env.adj)
    # print(sum(flags))
    # data = np.load('data.npy')

    # print(len(np.where(np.array(rewards) == 0)[0]))
    # print(rewards)
