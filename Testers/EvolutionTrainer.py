import pygame


class EvolutionTrainer():
    def __init__(self, driver, size, verbosity=1, error_length=100, error_bars=2):
        self.verbosity = verbosity
        self.driver = driver
        self.width = size[0]
        self.height = size[1]
        self.error_length = error_length
        self.error_bars = error_bars
        if size:
            pygame.init()
            self.screen = pygame.display.set_mode(size)
            pygame.key.set_repeat(100, 50)
            self.screen.fill([150, 150, 150])
        self.loop = True
        self.delay = 0
        self.testing = False

        self.error_array = []

    def main(self):
        while self.loop:
            pygame.time.delay(self.delay)

            if self.testing:
                self.driver.test()
                self.driver.draw(self.screen, 0, 0, self.width, self.height - 200, 10)
            else:
                self.driver.run_visual(self.screen, 0, 0, self.width, self.height - 200, 2)
            self.error_array.append((self.driver.Maximum, self.driver.Average, self.driver.Median, self.driver.Minimum))
            if len(self.error_array) > self.error_length:
                self.error_array.pop(0)

            self.screen.fill([150, 150, 150])
            if self.verbosity >= 1:
                self.driver.draw(self.screen, 0, 0, self.width, self.height - 200, 2)
                pygame.draw.line(self.screen, [0, 0, 0], [0, self.height], [self.width, self.height], 5)
                for i in range(1, self.error_bars):
                    pygame.draw.line(self.screen, [200, 200, 200], [0, self.height - i * (200 / self.error_bars)], [self.width, self.height - i * (200 / self.error_bars)], 5)
                pygame.draw.line(self.screen, [0, 0, 0], [0, self.height - 200], [self.width, self.height - 200], 5)
                print(self.error_array[-1][0])
                for i in range(len(self.error_array)):
                    pygame.draw.circle(self.screen, [255, 0, 0],
                                       [int(10 + (self.width - 10) * (i) / len(self.error_array)),
                                        int(self.height - (200 / self.error_bars) * self.error_array[i][3])], 5)
                    pygame.draw.circle(self.screen, [0, 255, 0],
                                       [int(10 + (self.width - 10) * (i) / len(self.error_array)),
                                        int(self.height - (200 / self.error_bars) * self.error_array[i][2])], 5)
                    pygame.draw.circle(self.screen, [0, 0, 255],
                                       [int(10 + (self.width - 10) * (i) / len(self.error_array)),
                                        int(self.height - (200 / self.error_bars) * self.error_array[i][1])], 5)
                    pygame.draw.circle(self.screen, [0, 0, 0],
                                       [int(10 + (self.width - 10) * (i) / len(self.error_array)),
                                        int(self.height - (200 / self.error_bars) * self.error_array[i][0])], 5)
            else:
                self.driver.draw(self.screen, 0, 0, self.width, self.height, 2)

            pygame.display.flip()
            self.handle_events()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.loop = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    if self.delay == 0:
                        self.delay = 500
                    else:
                        self.delay = 0
                elif event.key == pygame.K_SPACE:
                    self.testing = not self.testing
