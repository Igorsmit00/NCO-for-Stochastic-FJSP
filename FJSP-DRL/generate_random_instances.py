import os

import numpy as np
import scipy
from tqdm import tqdm

from data_utils import load_data_from_single_file, matrix_to_text

NUM_SAMPLES = 5000


def get_all_files(directory: str):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def create_stochastic_realizations(original_instance, num_samples):
    # Assuming duration is your numpy array of shape (50,5)
    duration = original_instance
    # Create a mask for non-zero values
    mask = duration != 0
    # Calculate mean and sigma only for non-zero values
    mean = np.zeros_like(duration, dtype=np.float64)
    sigma = np.zeros_like(duration, dtype=np.float64)

    mean[mask] = np.log(duration[mask])
    sigma[mask] = np.sqrt(
        np.log(1 + np.square(duration[mask] * 0.25) / np.square(duration[mask]))
    )
    tile_mean = np.tile(mean, (num_samples, 1, 1))
    tile_sigma = np.tile(sigma, (num_samples, 1, 1))
    tile_mask = np.tile(mask, (num_samples, 1, 1))
    # Generate samples for each item in duration
    samples = np.zeros_like(tile_mean, dtype=np.int64)
    samples[tile_mask] = np.maximum(
        np.round(
            np.random.lognormal(mean=tile_mean[tile_mask], sigma=tile_sigma[tile_mask])
        ),
        1,
    )
    return samples


def create_stochastic_realizations_random_variances(original_instance, num_samples):
    # Assuming duration is your numpy array of shape (50,5)
    duration = original_instance
    # Create a mask for non-zero values
    mask = duration != 0
    # Calculate mean and sigma only for non-zero values
    mean = np.zeros_like(duration, dtype=np.float64)
    sigma = np.zeros_like(duration, dtype=np.float64)
    mean[mask] = np.log(duration[mask])
    CoV = np.random.uniform(0.1, 0.5, size=duration.shape)
    sigma[mask] = np.sqrt(
        np.log(1 + np.square(duration[mask] * CoV[mask]) / np.square(duration[mask]))
    )

    tile_mean = np.tile(mean, (num_samples, 1, 1))
    tile_sigma = np.tile(sigma, (num_samples, 1, 1))
    tile_mask = np.tile(mask, (num_samples, 1, 1))
    # Generate samples for each item in duration
    samples = np.zeros_like(tile_mean, dtype=np.int64)
    samples[tile_mask] = np.maximum(
        np.round(
            np.random.lognormal(mean=tile_mean[tile_mask], sigma=tile_sigma[tile_mask])
        ),
        1,
    )
    return samples


def create_stochastic_realizations_random_beta(original_instance, num_samples):
    # Assuming duration is your numpy array of shape (50,5)
    duration = original_instance
    # Create a mask for non-zero values
    mask = duration != 0
    c = np.zeros_like(duration, dtype=np.float64)
    d = np.zeros_like(duration, dtype=np.float64)
    alpha = np.zeros_like(duration, dtype=np.float64)
    beta = np.zeros_like(duration, dtype=np.float64)
    sigma = np.zeros_like(duration, dtype=np.float64)
    c[mask] = duration[mask] - 0.2 * duration[mask]
    d[mask] = duration[mask] + 0.8 * duration[mask]
    CoV = np.random.uniform(0.1, 0.5, size=duration.shape)
    sigma[mask] = duration[mask] * CoV[mask]
    alpha[mask] = (
        np.square((duration[mask] - c[mask]) / (d[mask] - c[mask]))
        * (1 - (duration[mask] - c[mask]) / (d[mask] - c[mask]))
    ) / (np.square(sigma[mask]) / np.square(d[mask] - c[mask])) - (
        (duration[mask] - c[mask]) / (d[mask] - c[mask])
    )
    beta[mask] = (
        ((duration[mask] - c[mask]) / (d[mask] - c[mask]))
        * (1 - ((duration[mask] - c[mask]) / (d[mask] - c[mask])))
        / (np.square(sigma[mask]) / np.square(d[mask] - c[mask]))
        - alpha[mask]
        - 1
    )
    invalid_mask = ((alpha <= 0) | (beta <= 0)) & mask
    alpha[invalid_mask] = 13
    beta[invalid_mask] = 26
    c[invalid_mask] = np.maximum(1, duration[invalid_mask] - 0.5)
    d[invalid_mask] = duration[invalid_mask] + 2.5
    tile_c = np.tile(c, (num_samples, 1, 1))
    tile_d = np.tile(d, (num_samples, 1, 1))
    tile_alpha = np.tile(alpha, (num_samples, 1, 1))
    tile_beta = np.tile(beta, (num_samples, 1, 1))
    tile_mask = np.tile(mask, (num_samples, 1, 1))
    samples = np.zeros_like(tile_alpha, dtype=np.int64)
    samples[tile_mask] = np.maximum(
        np.round(
            scipy.stats.beta.rvs(
                tile_alpha[tile_mask],
                tile_beta[tile_mask],
                loc=tile_c[tile_mask],
                scale=tile_d[tile_mask] - tile_c[tile_mask],
            )
        ),
        1,
    )
    return samples


def create_stochastic_realizations_random_beta_log_mix(original_instance, num_samples):
    # Assuming duration is your numpy array of shape (50,5)
    duration = original_instance
    # Create a mask for non-zero values
    mask = duration != 0

    CoV = np.random.uniform(0.1, 0.5, size=duration.shape)

    mask_beta_start = np.random.binomial(1, 0.5, size=duration.shape).astype(bool)
    mask_beta = mask & mask_beta_start
    mask_log = mask & ~mask_beta_start
    ##### LOGNORMAL
    mean = np.zeros_like(duration, dtype=np.float64)
    sigma = np.zeros_like(duration, dtype=np.float64)
    mean[mask_log] = np.log(duration[mask_log])
    sigma[mask_log] = np.sqrt(
        np.log(
            1
            + np.square(duration[mask_log] * CoV[mask_log])
            / np.square(duration[mask_log])
        )
    )

    tile_mean = np.tile(mean, (num_samples, 1, 1))
    tile_sigma = np.tile(sigma, (num_samples, 1, 1))
    tile_mask_log = np.tile(mask_log, (num_samples, 1, 1))
    # Generate samples for each item in duration
    samples = np.zeros_like(tile_mean, dtype=np.int64)
    samples[tile_mask_log] = np.maximum(
        np.round(
            np.random.lognormal(
                mean=tile_mean[tile_mask_log], sigma=tile_sigma[tile_mask_log]
            )
        ),
        1,
    )

    ##### BETA

    c = np.zeros_like(duration, dtype=np.float64)
    d = np.zeros_like(duration, dtype=np.float64)
    alpha = np.zeros_like(duration, dtype=np.float64)
    beta = np.zeros_like(duration, dtype=np.float64)
    sigma = np.zeros_like(duration, dtype=np.float64)
    c[mask_beta] = duration[mask_beta] - 0.2 * duration[mask_beta]
    d[mask_beta] = duration[mask_beta] + 0.8 * duration[mask_beta]
    CoV = np.random.uniform(0.1, 0.5, size=duration.shape)
    sigma[mask_beta] = duration[mask_beta] * CoV[mask_beta]
    alpha[mask_beta] = (
        np.square((duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta]))
        * (1 - (duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta]))
    ) / (np.square(sigma[mask_beta]) / np.square(d[mask_beta] - c[mask_beta])) - (
        (duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta])
    )
    beta[mask_beta] = (
        ((duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta]))
        * (1 - ((duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta])))
        / (np.square(sigma[mask_beta]) / np.square(d[mask_beta] - c[mask_beta]))
        - alpha[mask_beta]
        - 1
    )
    invalid_mask = ((alpha <= 0) | (beta <= 0)) & mask_beta
    alpha[invalid_mask] = 13
    beta[invalid_mask] = 26
    c[invalid_mask] = np.maximum(1, duration[invalid_mask] - 0.5)
    d[invalid_mask] = duration[invalid_mask] + 2.5
    tile_c = np.tile(c, (num_samples, 1, 1))
    tile_d = np.tile(d, (num_samples, 1, 1))
    tile_alpha = np.tile(alpha, (num_samples, 1, 1))
    tile_beta = np.tile(beta, (num_samples, 1, 1))
    tile_mask = np.tile(mask_beta, (num_samples, 1, 1))
    samples[tile_mask] = np.maximum(
        np.round(
            scipy.stats.beta.rvs(
                tile_alpha[tile_mask],
                tile_beta[tile_mask],
                loc=tile_c[tile_mask],
                scale=tile_d[tile_mask] - tile_c[tile_mask],
            )
        ),
        1,
    )

    return samples


def create_stochastic_realizations_random_beta_log_gamma_mix(
    original_instance, num_samples
):
    # Assuming duration is your numpy array of shape (50,5)
    duration = original_instance
    # Create a mask for non-zero values
    mask = duration != 0

    CoV = np.random.uniform(0.1, 0.5, size=duration.shape)

    dist = np.random.choice(3, size=duration.shape)
    mask_beta = mask & (dist == 0)
    mask_log = mask & (dist == 1)
    mask_gamma = mask & (dist == 2)
    ##### LOGNORMAL
    mean = np.zeros_like(duration, dtype=np.float64)
    sigma = np.zeros_like(duration, dtype=np.float64)
    mean[mask_log] = np.log(duration[mask_log])
    sigma[mask_log] = np.sqrt(
        np.log(
            1
            + np.square(duration[mask_log] * CoV[mask_log])
            / np.square(duration[mask_log])
        )
    )

    tile_mean = np.tile(mean, (num_samples, 1, 1))
    tile_sigma = np.tile(sigma, (num_samples, 1, 1))
    tile_mask_log = np.tile(mask_log, (num_samples, 1, 1))
    # Generate samples for each item in duration
    samples = np.zeros_like(tile_mean, dtype=np.int64)
    samples[tile_mask_log] = np.maximum(
        np.round(
            np.random.lognormal(
                mean=tile_mean[tile_mask_log], sigma=tile_sigma[tile_mask_log]
            )
        ),
        1,
    )

    ##### BETA

    c = np.zeros_like(duration, dtype=np.float64)
    d = np.zeros_like(duration, dtype=np.float64)
    alpha = np.zeros_like(duration, dtype=np.float64)
    beta = np.zeros_like(duration, dtype=np.float64)
    sigma = np.zeros_like(duration, dtype=np.float64)
    c[mask_beta] = duration[mask_beta] - 0.2 * duration[mask_beta]
    d[mask_beta] = duration[mask_beta] + 0.8 * duration[mask_beta]

    sigma[mask_beta] = duration[mask_beta] * CoV[mask_beta]
    alpha[mask_beta] = (
        np.square((duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta]))
        * (1 - (duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta]))
    ) / (np.square(sigma[mask_beta]) / np.square(d[mask_beta] - c[mask_beta])) - (
        (duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta])
    )
    beta[mask_beta] = (
        ((duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta]))
        * (1 - ((duration[mask_beta] - c[mask_beta]) / (d[mask_beta] - c[mask_beta])))
        / (np.square(sigma[mask_beta]) / np.square(d[mask_beta] - c[mask_beta]))
        - alpha[mask_beta]
        - 1
    )
    invalid_mask = ((alpha <= 0) | (beta <= 0)) & mask_beta
    alpha[invalid_mask] = 13
    beta[invalid_mask] = 26
    c[invalid_mask] = np.maximum(1, duration[invalid_mask] - 0.5)
    d[invalid_mask] = duration[invalid_mask] + 2.5
    tile_c = np.tile(c, (num_samples, 1, 1))
    tile_d = np.tile(d, (num_samples, 1, 1))
    tile_alpha = np.tile(alpha, (num_samples, 1, 1))
    tile_beta = np.tile(beta, (num_samples, 1, 1))
    tile_mask = np.tile(mask_beta, (num_samples, 1, 1))
    samples[tile_mask] = np.maximum(
        np.round(
            scipy.stats.beta.rvs(
                tile_alpha[tile_mask],
                tile_beta[tile_mask],
                loc=tile_c[tile_mask],
                scale=tile_d[tile_mask] - tile_c[tile_mask],
            )
        ),
        1,
    )

    k = np.zeros_like(duration, dtype=np.float64)
    theta = np.zeros_like(duration, dtype=np.float64)
    k[mask_gamma] = np.square(duration[mask_gamma]) / np.square(
        duration[mask_gamma] * CoV[mask_gamma]
    )
    theta[mask_gamma] = (
        np.square(duration[mask_gamma] * CoV[mask_gamma]) / duration[mask_gamma]
    )
    tile_k = np.tile(k, (num_samples, 1, 1))
    tile_theta = np.tile(theta, (num_samples, 1, 1))
    tile_mask_gamma = np.tile(mask_gamma, (num_samples, 1, 1))
    samples[tile_mask_gamma] = np.maximum(
        np.round(
            np.random.gamma(
                shape=tile_k[tile_mask_gamma], scale=tile_theta[tile_mask_gamma]
            )
        ),
        1,
    )

    return samples


# Use the following code below if you want to generate random instance for the Benchmark data
if __name__ == "__main__":
    from params import configs

    train_vali_str = "/data_train_vali" if configs.data_type == "vali" else ""
    directory = f"./data{train_vali_str}/{configs.data_source}/{configs.test_data[0]}"
    print(directory)
    for original_instance in tqdm(get_all_files(directory), total=100):
        job_length, processing_times = load_data_from_single_file(original_instance)
        if configs.variance == "random":
            if configs.variance_dist == "beta":
                processing_times = create_stochastic_realizations_random_beta(
                    processing_times, NUM_SAMPLES
                )
            elif configs.variance_dist == "be_lo":
                processing_times = create_stochastic_realizations_random_beta_log_mix(
                    processing_times, NUM_SAMPLES
                )
            elif configs.variance_dist == "be_lo_ga":
                processing_times = (
                    create_stochastic_realizations_random_beta_log_gamma_mix(
                        processing_times, NUM_SAMPLES
                    )
                )
            elif configs.variance_dist == "lognormal":
                processing_times = create_stochastic_realizations_random_variances(
                    processing_times, NUM_SAMPLES
                )
        elif configs.variance == "fixed":
            processing_times = create_stochastic_realizations(
                processing_times, NUM_SAMPLES
            )
        or_instance_name = original_instance.split("\\")[-1].split(".")[0]
        dist_str = (
            f"{configs.variance_dist}/" if configs.variance_dist != "lognormal" else ""
        )
        realization_file = f"./data{train_vali_str}/stochastic_realizations/{configs.variance}/{dist_str}{configs.data_source}/{configs.test_data[0]}/{or_instance_name}.fjs"
        realization_dir = realization_file.removesuffix(f"/{or_instance_name}.fjs")
        if not os.path.exists(realization_dir):
            os.makedirs(realization_dir)

        with open(original_instance, "r") as f_or, open(
            realization_file, "w"
        ) as f_real:
            for line in f_or:
                f_real.write(line)
            for i in range(NUM_SAMPLES):
                f_real.write("\n")
                pt = processing_times[i]
                lines_doc = matrix_to_text(job_length, pt, "", first_line=False)
                for j in range(len(lines_doc)):
                    f_real.write(lines_doc[j] + "\n")
