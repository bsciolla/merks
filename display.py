
import os
import pygame
from pygame.compat import geterror

from pygame.locals import KEYDOWN, \
    RLEACCEL, K_ESCAPE, QUIT, MOUSEBUTTONDOWN, MOUSEBUTTONUP, K_q
from pygame.transform import rotate

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, 'data')

from globalvars import WIN_X, WIN_Y, MERKS_NUM, CLOCK_FPS

import numpy
import world
import random
random.seed(112)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def load_image(name, colorkey=None):
    fullname = os.path.join(data_dir, name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error:
        print('Cannot load image:', fullname)
        raise SystemExit(str(geterror()))
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Merksprite(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image('merk2.bmp', -1)
        self.original = self.image

    # ------------------------------------------------ #

    def update(self, x, y, angle):
        self.rect.midtop = (x, y)
        self.image = rotate(self.original, -angle*180.0/numpy.pi)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# def main():
try:
    thisworld = world.World(merks_num=MERKS_NUM)

    pygame.init()
    screen = pygame.display.set_mode((WIN_X, WIN_Y))
    pygame.display.set_caption('World of merks')
    pygame.mouse.set_visible(1)

    
#showmap = thisworld.stage_.displaymap()
        #snapshot = pygame.surfarray.make_surface(showmap)
    #screen.blit(snapshot, (0, 0))
    pygame.display.flip()
    clock = pygame.time.Clock()

    spritelist = []
    for i in range(MERKS_NUM):
        spritelist.append(Merksprite())

    allsprites = pygame.sprite.RenderPlain(spritelist)

    going = True

    while going:
        clock.tick(CLOCK_FPS)

        for event in pygame.event.get():
            if event.type == QUIT:
                going = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                going = False
            elif event.type == KEYDOWN and event.key == K_q:
                going = False
            elif event.type == MOUSEBUTTONDOWN:
                going = True

        thisworld.step_forward()
        world.update_sprites(spritelist, thisworld)
        showmap = thisworld.stage_.displaymap()
        snapshot = pygame.surfarray.make_surface(showmap)
        screen.blit(snapshot, (0, 0))
        allsprites.draw(screen)

        pygame.display.flip()
finally:
    pygame.quit()


# if __name__ == '__main__':
    # main()
