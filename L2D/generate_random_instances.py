import numpy as np

from Params import configs


def create_stochastic_realizations_random_variances(original_instance, num_samples):
    processing_times = original_instance[0]
    mean = np.log(processing_times)
    CoV = np.random.uniform(0.1, 0.5, size=processing_times.shape)
    sigma = np.sqrt(
        np.log(1 + np.square(processing_times * CoV) / np.square(processing_times))
    )
    tile_mean = np.tile(mean, (num_samples, 1, 1))
    tile_sigma = np.tile(sigma, (num_samples, 1, 1))
    samples = np.maximum(
        np.round(np.random.lognormal(mean=tile_mean, sigma=tile_sigma)),
        1,
    ).astype(np.int32)
    return samples


if __name__ == "__main__":
    dataLoaded = np.load(
        "./DataGen/generatedData"
        + str(configs.n_j)
        + "_"
        + str(configs.n_m)
        + "_Seed"
        + str(configs.np_seed_validation)
        + ".npy"
    )
    or_data = []
    for i in range(dataLoaded.shape[0]):
        or_data.append((dataLoaded[i][0], dataLoaded[i][1]))

    num_samples = 1100
    all_samples = []
    for original_instance in or_data:
        all_samples.append(
            create_stochastic_realizations_random_variances(
                original_instance, num_samples
            )
        )
    all_samples = np.stack(all_samples)
    np.save(
        "./DataGen/stochastic/generatedData"
        + str(configs.n_j)
        + "_"
        + str(configs.n_m)
        + "_Seed"
        + str(configs.np_seed_validation)
        + ".npy",
        all_samples,
    )
