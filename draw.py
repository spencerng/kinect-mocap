import cv2 as cv
import pygame as pg

from kinect import KinectCam

# Remember to run with sudo on linux

CAPTURE = True
FILENAME = "child1"


def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pg.image.frombuffer(image.tostring(), image.shape[1::-1], "BGR")

def main():
    pg.display.init()
    pg.font.init()
    pg.display.set_caption("Motion Capture")

    screen = pg.display.set_mode((1920, 1080), pg.RESIZABLE)
    clock = pg.time.Clock()

    cam = KinectCam(CAPTURE, FILENAME)

    running = True
    

    centroid_window = list()

    game_running_timer = 0

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        cv_img, centroids, player_contour = cam.get_frame()

        screen.fill((0, 0, 0))

        if cv_img is not None:

            if player_contour is not None:
                screen.blit(cvimage_to_pygame(cv_img), (0, 0))
                # pg.draw.polygon(screen, (0, 255, 0), player_contour)
                # pg.draw.circle(screen, (255, 0, 0), (x_med, y_med), 30)

        game_running_timer += 1
        pg.display.update()

        # Limit FPS
        clock.tick(30)

    cam.close()


if __name__ == "__main__":
    main()
