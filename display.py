import pygame
import numpy as np
from pygame.locals import *
from nn.activation_functions import *
from nn.layers import Dense, InLayer
from nn.nets import Net
from nn.cost_functions import *
import matplotlib.pyplot as plt

class grid():
    def __init__(self, shape:tuple[int, int], display_shape:tuple[int, int], topleft:tuple[int, int]):
        self._shape = shape
        self.grid = np.zeros(shape)
        self.display_shape = display_shape
        self.topleft = topleft

    def add_point(self, real_pos:tuple[int, int]):
        # Convert real (screen) position to grid coordinates
        grid_x = (real_pos[0] - self.topleft[0]) * (self._shape[0] / self.display_shape[0])
        grid_y = (real_pos[1] - self.topleft[1]) * (self._shape[1] / self.display_shape[1])

        if 0 <= grid_x < self._shape[0] and 0 <= grid_y < self._shape[1]:
            # Create meshgrid of coordinates
            x_indices, y_indices = np.indices(self._shape)
            
            # Compute squared Euclidean distance from the clicked point
            dist_sq = (x_indices - grid_x) ** 2 + (y_indices - grid_y) ** 2

            # Example: apply Gaussian brightness falloff
            sigma = 0.65  # adjust for spread
            brightness = np.exp(-dist_sq / (2 * sigma ** 2))

            # Add to existing grid, clip to max brightness of 1
            self.grid = np.clip(self.grid + brightness, 0, 1)

    def draw_grid(self, screen):
        cell_w = int(round(self.display_shape[0]/self._shape[0]))
        cell_h = int(round(self.display_shape[1]/self._shape[1]))
        for y in range(self._shape[1]):
            for x in range(self._shape[0]):
                rect = pygame.Rect(x * cell_w + self.topleft[0], y * cell_h + self.topleft[1], cell_w, cell_h)
                val = self.grid[x, y]*255
                color = (val, val, val)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)  # Grid lines
 
class App:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self.size = self.width, self.height = 720, 400
        self.grid = grid(shape=(28, 28), display_shape=(300, 300), topleft=(20, 20))
        self.net = Net.load_model('nn/model.npz')
        self.clock = pygame.time.Clock()
 
    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
        self.grid.draw_grid(self._display_surf)
        
        
 
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        if (event.type == pygame.MOUSEBUTTONDOWN 
            or (event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0])):
            pos = pygame.mouse.get_pos()
            self.grid.add_point(pos)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # replace with any key
                print("Space key pressed")
                self.grid.grid = np.zeros(self.grid._shape)

    def on_loop(self):
        pass
    def on_render(self):
        self._display_surf.fill((0, 0, 0))

        self.grid.draw_grid(self._display_surf)

        X = self.grid.grid.T.reshape((1,-1))
        Y_pred = self.net.predict(X)
        self.show_bar_graph(Y_pred[0])

        pygame.display.flip()
    def on_cleanup(self):
        pygame.quit()

    def show_bar_graph(self, y_pred):
        font = pygame.font.SysFont(None, 30)
        font2 = pygame.font.SysFont(None, 50)
        bar_height = 30
        spacing = 4
        top = 10
        left = 500
        y_pred_val = np.argmax(y_pred)
        self._display_surf.blit(font2.render(f'{y_pred_val}', True, (255, 255, 255)), (left, top + 350))
        for i, value in enumerate(y_pred):
            y = i * (bar_height + spacing)
            text_surface = font.render(f'{y_pred[i]:5.3f}', True, (255, 255, 255))
            self._display_surf.blit(text_surface, (left - 70, top + y + bar_height/2 - 12))
            pygame.draw.rect(self._display_surf , (255, 255, 255), (left, top + y,value*100, bar_height))
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
            self.clock.tick(80)
        self.on_cleanup()
 
if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()