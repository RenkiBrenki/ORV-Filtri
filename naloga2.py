import cv2 as cv
import numpy as np

import numpy as np
import cv2 as cv


def konvolucija_barvna(image, kernel):
    '''Izvede konvolucijo nad sliko. Brez uporabe funkcije cv.filter2D, ali katerekoli druge funkcije, ki izvaja konvolucijo.
    Funkcijo implementirajte sami z uporabo zank oz. vektorskega računanja.'''
    kernel_height, kernel_width = kernel.shape
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)),
                          mode='reflect')

    image_height, image_width, channels = image.shape
    return_image = np.zeros_like(image)

    for c in range(channels):
        for i in range(image_height):
            for j in range(image_width):
                # Extract the image patch
                image_patch = padded_image[i:i + kernel_height, j:j + kernel_width, c]

                # Compute the convolution
                result = np.sum(image_patch * kernel)

                # Save the result to the corresponding pixel in the output image
                return_image[i, j, c] = result

    return return_image[:image_height, :image_width]

def konvolucija(image, kernel):
    kernel_height, kernel_width = kernel.shape

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    padded_image = np.pad(image, ((padding_height, padding_width), (padding_height, padding_width)), mode='reflect')

    image_height, image_width = image.shape
    return_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            # Damo jedro cez sliko patch = del slike
            image_patch = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Zmnozimo dobljeno matriko z jedrom
            result = np.sum(image_patch * kernel)

            # Piksel shranimo na ustrezno mesto v sliki
            return_image[i, j] = result

    return return_image[:image_height, :image_width]

def filtriraj_z_gaussovim_jedrom(slika, sigma=float):
    '''Filtrira sliko z Gaussovim jedrom..'''
    kernel_size = int(2 * sigma) * 2 + 1
    k = kernel_size / 2 - 1 / 2

    kernel = np.zeros((kernel_size, kernel_size))

    constant = 1 / (2 * np.pi * sigma ** 2)

    for i in range(kernel_size):
        for j in range(kernel_size):
            exponent = - (((i - k - 1) ** 2 + (j - k - 1) ** 2) / (2 * sigma ** 2))
            kernel[i, j] = constant * np.exp(exponent)

    return_image = konvolucija(slika, kernel)
    return return_image


def filtriraj_sobel_smer(slika):
    '''Filtrira sliko z Sobelovim jedrom in označi gradiente v orignalni sliki glede na ustrezen pogoj.'''
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    sobel_filtered = filtriraj_z_gaussovim_jedrom(slika, 0.5)

    gradient_x = konvolucija(sobel_filtered, sobel_x)
    gradient_y = konvolucija(sobel_filtered, sobel_y)

    sobel = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    sobel = np.uint8(sobel)

    return sobel

def process_sobel_image(image):
    output = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= 150:
                output[i, j] = 0, 255, 0
            else:
                output[i, j] = 0, 0, 0

    return output

if __name__ == '__main__':
    image = cv.imread(".utils/lenna.png")
    cv.imshow("original", image)

    image = np.float64(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

    #image = filtriraj_z_gaussovim_jedrom(image, 2.0)
    image = filtriraj_sobel_smer(image)

    cv.imshow("Sobel", image)

    image = process_sobel_image(image)

    cv.imshow("Gradient", image)

    cv.waitKey(0)
    cv.destroyAllWindows()
    pass
