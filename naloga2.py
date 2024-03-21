import cv2 as cv
import numpy as np

import numpy as np
import cv2 as cv


def konvolucija(image, kernel):
    '''Izvede konvolucijo nad sliko. Brez uporabe funkcije cv.filter2D, ali katerekoli druge funkcije, ki izvaja konvolucijo.
    Funkcijo implementirajte sami z uporabo zank oz. vektorskega računanja.'''
    kernel_height, kernel_width = kernel.shape
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)), mode='constant')

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

    pass

def zmanjsaj_sliko(slika, sirina, visina):
    '''Zmanjšaj sliko na velikost sirina x visina.'''
    return cv.resize(slika, (sirina, visina))

if __name__ == '__main__':

    image = cv.imread(".utils/lenna.png")
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)/9
    kernel2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32) #Corner detection

    cv.imshow("original", image)

    image = filtriraj_z_gaussovim_jedrom(image, 1.0)

    cv.imshow("Gaussian blur", image)

    cv.waitKey(0)
    cv.destroyAllWindows()

    #camera = cv.VideoCapture(1)
    # while True:
    #     ret, frame = camera.read()
    #
    #     #frame = zmanjsaj_sliko(frame, 300, 260)
    #
    #     kernel1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)  #Box blur
    #     #kernel2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32) #Corner detection
    #
    #     frame = konvolucija(frame, kernel1)
    #
    #     cv.imshow("Camera", frame)
    #
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # camera.release()
    # cv.destroyAllWindows()
