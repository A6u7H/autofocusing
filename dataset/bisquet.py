import numpy as np
import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize
import collections
import cv2

from itertools import chain


def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) 
    return normalize_kernel(gaussian_kernel)


def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')


def local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)
    return (image - local_mean) / (local_var + C)


def generalized_gaussian_dist(x, alpha, sigma):
    beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))

    coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)


def calculate_pair_product_coefficients(mscn_coefficients):
    return collections.OrderedDict({
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })


def asymmetric_generalized_gaussian(x, nu, sigma_l, sigma_r):
    def beta(sigma):
        return sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))

    coefficient = nu / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu))
    f = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu)

    return np.where(
        x < 0,
        f(-x, sigma_l),
        f(x, sigma_r)
    )


def asymmetric_generalized_gaussian_fit(x):
    def estimate_phi(alpha):
        numerator = special.gamma(2 / alpha) ** 2
        denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

    def estimate_R_hat(r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(
            x,
            filter=lambda z: z == z
    ):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))

    def estimate_gamma(x):
        left_squares = mean_squares_sum(x, lambda z: z < 0)
        right_squares = mean_squares_sum(x, lambda z: z >= 0)

        return np.sqrt(left_squares) / np.sqrt(right_squares)

    def estimate_alpha(x):
        r_hat = estimate_r_hat(x)
        gamma = estimate_gamma(x)
        R_hat = estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x

        return solution[0]

    def estimate_sigma(
            x,
            alpha,
            filter = lambda z: z < 0
    ):
        return np.sqrt(mean_squares_sum(x, filter))

    def estimate_mean(alpha, sigma_l, sigma_r):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))

    alpha = estimate_alpha(x)
    sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
    sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)

    constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    mean = estimate_mean(alpha, sigma_l, sigma_r)

    return alpha, mean, sigma_l, sigma_r


def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
    def calculate_features(
            coefficients_name,
            coefficients
    ):
        alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(
            coefficients
        )
        if coefficients_name == 'mscn':
            var = (sigma_l ** 2 + sigma_r ** 2) / 2
            return [alpha, var]
        return [alpha, mean, sigma_l ** 2, sigma_r ** 2]

    mscn_coefficients = calculate_mscn_coefficients(image, kernel_size, sigma)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)

    features = [
        calculate_features(name, coeff)
        for name, coeff in coefficients.items()
    ]
    flatten_features = list(chain.from_iterable(features))
    return flatten_features


def get_besquet_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    downscaled_image = cv2.resize(
        gray,
        None,
        fx=1/2,
        fy=1/2,
        interpolation=cv2.INTER_CUBIC
    )

    downscale_brisque_features = calculate_brisque_features(
        downscaled_image,
        kernel_size=7,
        sigma=7/6
    )
    brisque_features = calculate_brisque_features(
        gray,
        kernel_size=7,
        sigma=7/6
    )

    downscale_brisque_features = [float(feature) for feature in downscale_brisque_features]
    brisque_features = [float(feature) for feature in brisque_features]

    brisque_features = np.concatenate(
        [brisque_features, downscale_brisque_features], dtype=np.float32
    )

    return brisque_features
