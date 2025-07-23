import pygame
import numpy as np
from pygame.locals import *
from nn.activation_functions import *
from nn.nets import Net
from nn.cost_functions import *

class grid():
    def __init__(self, shape:tuple[int, int], display_shape:tuple[int, int], topleft:tuple[int, int]):
        self._shape = shape
        self.grid = np.zeros(shape)
        self.grid_processed = np.zeros(shape)
        self.display_shape = display_shape
        self.topleft = topleft
        self.point_list = []
        self.scaled_pts_list = []

    def add_point(self, real_pos:tuple[int, int]):
        # Convert real (screen) position to grid coordinates
        grid_x = (real_pos[0] - self.topleft[0]) * (self._shape[0] / self.display_shape[0])
        grid_y = (real_pos[1] - self.topleft[1]) * (self._shape[1] / self.display_shape[1])

        if 0 <= grid_x < self._shape[0] and 0 <= grid_y < self._shape[1]:

            self.point_list.append([grid_x, grid_y])

    def render_grids(self):
        def draw_line(p1, p2, num=20):
            return np.linspace(p1, p2, num=num)

        def render_grid(point_list, shape):
            grid = np.zeros(shape)
            sigma = 0.45
            x_idx, y_idx = np.indices(shape)

            prev = None
            for pt in point_list:
                if pt is None:
                    prev = None  # Start of new stroke
                    continue
                if prev is not None:
                    for x, y in draw_line(prev, pt):
                        dist_sq = (x_idx - x) ** 2 + (y_idx - y) ** 2
                        brightness = np.exp(-dist_sq / (2 * sigma ** 2))
                        grid = np.clip(grid + brightness, 0, 1)
                prev = pt
            return grid

        self.grid = render_grid(self.point_list, self._shape)
        self.grid_processed = render_grid(self.scaled_pts_list, self._shape)
        
    def process_grid_coords(self):
        if not self.point_list:
            self.scaled_pts_list = []
            return

        # Extract valid points (ignore None for scaling)
        coords_only = np.array([pt for pt in self.point_list if pt is not None])
        
        if len(coords_only) == 0:
            self.scaled_pts_list = []
            return

        # Get bounding box
        min_xy = np.min(coords_only, axis=0)
        max_xy = np.max(coords_only, axis=0)
        size = max_xy - min_xy

        if np.any(size == 0):
            scaled = coords_only
        else:
            scale = 15 / max(size)  # 28 - 9 margin
            scaled = (coords_only - min_xy) * scale

        # Center around (14, 14)
        center = (np.min(scaled, axis=0) + np.max(scaled, axis=0)) / 2
        # center = np.mean(scaled, axis=0)
        translation = np.array([14, 14.5]) - center

        # Create a mapping of original points to scaled points
        scaled_iter = iter((scaled + translation).tolist())
        output = []

        for pt in self.point_list:
            if pt is None:
                output.append(None)
            else:
                output.append(next(scaled_iter))

        self.scaled_pts_list = output

    def draw_grids(self, screen):
        def draw_grid(display_shape, shape, topLeft, grid):
            cell_w = int(round(display_shape[0]/shape[0]))
            cell_h = int(round(display_shape[1]/shape[1]))
            for y in range(shape[1]):
                for x in range(shape[0]):
                    rect = pygame.Rect(x * cell_w + topLeft[0], y * cell_h + topLeft[1], cell_w, cell_h)
                    val = grid[x, y]*255
                    color = (val, val, val)
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, (200, 200, 200), rect, 1)  # Grid lines

        draw_grid(self.display_shape, self._shape, self.topleft, self.grid)
        draw_grid(self.display_shape, self._shape, [20, 440], self.grid_processed)

    def clear_grids(self):
        self.grid = np.zeros(self._shape)
        self.grid_processed = np.zeros(self._shape)
        self.point_list = []
        self.scaled_pts_list = []
    
class App:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self.size = self.width, self.height = 720, 800
        self.grid = grid(shape=(28, 28), display_shape=(300, 300), topleft=(20, 20))
        self.net = Net.load_model('classification_models/3h.npz')
        self.clock = pygame.time.Clock()
 
    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
        self.grid.draw_grids(self._display_surf)
        
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            # Start of a new stroke
            self.grid.point_list.append(None)
            self.grid.add_point(pygame.mouse.get_pos())

        if (event.type == pygame.MOUSEBUTTONDOWN 
            or (event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0])):
            pos = pygame.mouse.get_pos()
            self.grid.add_point(pos)
            self.grid.process_grid_coords()
            self.grid.render_grids()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.grid.clear_grids()

    def on_loop(self):
        pass
    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        font = pygame.font.SysFont(None, 30)
        self._display_surf.blit(font.render(f'Pre-processing', True, (255, 255, 255)), (20, 400))
        self._display_surf.blit(font.render(f'Probability distribution', True, (255, 255, 255)), (400, 20))
        # self._display_surf.blit(font.render(f'y_pred = ', True, (255, 255, 255)), (20, 650))

        self.grid.draw_grids(self._display_surf)

        X = self.grid.grid_processed.T.reshape((1,-1))
        Y_pred = self.net.predict(X)
        self.show_bar_graph(Y_pred[0])

        pygame.display.flip()
    def on_cleanup(self):
        pygame.quit()

    def show_bar_graph(self, y_pred):
        font = pygame.font.SysFont(None, 30)
        font2 = pygame.font.SysFont(None, 70)
        bar_height = 50
        spacing = 15
        top = 50
        left = 500
        y_pred_val = np.argmax(y_pred)
        self._display_surf.blit(font2.render(f'y_pred = {y_pred_val}', True, (255, 255, 255)), (left - 100, top + 650))
        for i, value in enumerate(y_pred):
            y = i * (bar_height + spacing)
            text_surface = font.render(f'{i}:   {y_pred[i]:5.3f}', True, (255, 255, 255))
            self._display_surf.blit(text_surface, (left - 90, top + y + bar_height/2 - 12))
            pygame.draw.rect(self._display_surf , (255, 255, 255), (left + 10, top + y,value*100, bar_height))
 
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