import cv2 as cv
import numpy as np

def konvolucija(slika, jedro):
    '''Izvede konvolucijo nad sliko. Brez uporabe funkcije cv.filter2D, ali katerekoli druge funkcije, ki izvaja konvolucijo.
    Funkcijo implementirajte sami z uporabo zank oz. vektorskega računanja.'''
    pass

def filtriraj_z_gaussovim_jedrom(slika,sigma):
    '''Filtrira sliko z Gaussovim jedrom..'''
    pass

def filtriraj_sobel_smer(slika):
    '''Filtrira sliko z Sobelovim jedrom in označi gradiente v orignalni sliki glede na ustrezen pogoj.'''
    pass

if __name__ == '__main__':
    camera = cv.VideoCapture(1)
    while True:
        ret, frame = camera.read()
        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()
    pass
