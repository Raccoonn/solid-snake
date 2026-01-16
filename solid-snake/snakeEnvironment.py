from __future__ import annotations

import typing as t

import numpy as np


try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None  # type: ignore[assignment]


class snakeGame_v3:
    """
    Return state as 0/1 array of observations.

    Original behavior preserved from your version. :contentReference[oaicite:5]{index=5}
    """

    def __init__(self, screen_Width: int = 800, screen_Height: int = 800, N_sqrs: int = 25, difficulty: int = 25) -> None:
        """
        DESCRIPTION: Initialize the snake environment.

        PARAMETERS: screen_Width (OPT, int), by default 800 - Window width in pixels.
                    screen_Height (OPT, int), by default 800 - Window height in pixels.
                    N_sqrs (OPT, int), by default 25 - Grid size (N x N).
                    difficulty (OPT, int), by default 25 - FPS cap.

        RETURNS: None
        """
        self.screen_Width: int = int(screen_Width)
        self.screen_Height: int = int(screen_Height)
        self.N_sqrs: int = int(N_sqrs)
        self.difficulty: int = int(difficulty)

        self.dx: int = self.screen_Width // self.N_sqrs
        self.dy: int = self.screen_Height // self.N_sqrs

        # Runtime state (set in reset())
        self.direction: str = "RIGHT"
        self.snake_head: list[int] = [0, 0]
        self.snake_body: list[list[int]] = []
        self.last_len: int = 0
        self.food_pos: list[int] = [0, 0]
        self.score: int = 0

        # Rendering fields
        self.screen: t.Any = None
        self.clock: t.Any = None
        self.black: t.Any = None
        self.red: t.Any = None
        self.green: t.Any = None

    def state_observation(self) -> np.ndarray:
        """
        DESCRIPTION: Perform state observation, returning a 12-length feature vector.

        PARAMETERS: None

        RETURNS: np.ndarray - Shape (12,), dtype float.
        """
        state = np.zeros(12, dtype=np.float32)

        sx, sy = self.snake_head
        ax, ay = self.food_pos

        # Food locations (relative)
        if ay < sy:
            state[0] = 1.0
        if ax > sx:
            state[1] = 1.0
        if ay > sy:
            state[2] = 1.0
        if ax < sx:
            state[3] = 1.0

        # Obstacles (walls)
        if sx == 0:
            state[4] = 1.0
        if sy == self.N_sqrs - 1:
            state[5] = 1.0
        if sx == self.N_sqrs - 1:
            state[6] = 1.0
        if sy == 0:
            state[7] = 1.0

        # Body obstacles
        hx, hy = self.snake_head
        for seg in self.snake_body[3:]:
            if seg[0] == hx and seg[1] == hy - 1:
                state[4] = 1.0
            if seg[1] == hy and seg[0] == hx + 1:
                state[5] = 1.0
            if seg[0] == hx and seg[1] == hy + 1:
                state[6] = 1.0
            if seg[1] == hy and seg[0] == hx - 1:
                state[7] = 1.0

        # Direction one-hot
        if self.direction == "UP":
            state[8] = 1.0
        if self.direction == "RIGHT":
            state[9] = 1.0
        if self.direction == "DOWN":
            state[10] = 1.0
        if self.direction == "LEFT":
            state[11] = 1.0

        return state

    def reset(self) -> tuple[bool, float, np.ndarray]:
        """
        DESCRIPTION: Reset snake body, direction, spawn food, and return initial state.

        PARAMETERS: None

        RETURNS: tuple[bool, float, np.ndarray] - (done, reward, state)
        """
        self.direction = "RIGHT"
        self.snake_head = [4, 8]
        self.snake_body = [[4, 8], [4, 7], [4, 6]]
        self.last_len = len(self.snake_body)

        self.food_pos = [int(np.random.randint(5, self.N_sqrs)), int(np.random.randint(5, self.N_sqrs))]
        self.score = 0

        done = False
        reward = 0.0
        state = self.state_observation()
        return done, reward, state

    def step(self, action: int, frame: int, buffer: int = 500) -> tuple[bool, float, np.ndarray]:
        """
        DESCRIPTION: Take an action and advance the environment by one step.

        PARAMETERS: action (REQ, int) - 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
                    frame (REQ, int) - Frame counter for loop-break logic.
                    buffer (OPT, int), by default 500 - Loop-break interval.

        RETURNS: tuple[bool, float, np.ndarray] - (done, reward, next_state)
        """
        done = False
        reward = 0.0

        # prevent instantaneous reversal
        if action == 0 and self.direction != "DOWN":
            self.direction = "UP"
        if action == 1 and self.direction != "UP":
            self.direction = "DOWN"
        if action == 2 and self.direction != "RIGHT":
            self.direction = "LEFT"
        if action == 3 and self.direction != "LEFT":
            self.direction = "RIGHT"

        ax = abs(self.snake_head[0] - self.food_pos[0])
        ay = abs(self.snake_head[1] - self.food_pos[1])

        # move
        if self.direction == "UP":
            self.snake_head[1] -= 1
        if self.direction == "DOWN":
            self.snake_head[1] += 1
        if self.direction == "LEFT":
            self.snake_head[0] -= 1
        if self.direction == "RIGHT":
            self.snake_head[0] += 1

        ax_ = abs(self.snake_head[0] - self.food_pos[0])
        ay_ = abs(self.snake_head[1] - self.food_pos[1])

        if ax_ < ax or ay_ < ay:
            reward += 1.0
        elif ax_ > ax or ay_ > ay:
            reward -= 1.0

        self.snake_body.insert(0, list(self.snake_head))

        # eat food
        if self.snake_head[0] == self.food_pos[0] and self.snake_head[1] == self.food_pos[1]:
            reward += 10.0
            self.score += 1
            while True:
                self.food_pos = [int(np.random.randint(5, self.N_sqrs)), int(np.random.randint(5, self.N_sqrs))]
                if self.food_pos not in self.snake_body:
                    break
        else:
            self.snake_body.pop()

        # loop-break
        if frame % buffer == 0 and self.last_len == len(self.snake_body):
            done = True
            reward -= 300.0
        else:
            self.last_len = len(self.snake_body)

        # bounds
        if self.snake_head[0] < 0 or self.snake_head[0] == self.N_sqrs:
            done = True
            reward -= 100.0
        if self.snake_head[1] < 0 or self.snake_head[1] == self.N_sqrs:
            done = True
            reward -= 100.0

        # self collision
        for block in self.snake_body[1:]:
            if self.snake_head[0] == block[0] and self.snake_head[1] == block[1]:
                done = True
                reward -= 100.0

        state = self.state_observation()

        # pseudo lose condition to stop looping
        if state[4] == 1.0 and state[6] == 1.0:
            done = True
            reward -= 100.0
        if state[5] == 1.0 and state[7] == 1.0:
            done = True
            reward -= 100.0

        return done, reward, state

    def setup_window(self) -> None:
        """
        DESCRIPTION: Initialize pygame window + colors + clock.

        PARAMETERS: None

        RETURNS: None

        EXCEPTIONS: RuntimeError - If pygame is not installed.
        """
        if pygame is None:
            raise RuntimeError("pygame is not available. Install it: pip install pygame")

        pygame.init()
        pygame.display.set_caption("Snake_v3")
        self.screen = pygame.display.set_mode((self.screen_Width, self.screen_Height))

        self.black = pygame.Color(0, 0, 0)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)

        self.clock = pygame.time.Clock()

    def render(self) -> None:
        """
        DESCRIPTION: Render the grid, snake, and food.

        PARAMETERS: None

        RETURNS: None
        """
        if pygame is None:
            return

        # allow closing window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)

        pygame.display.flip()
        self.screen.fill(self.black)

        grid = np.zeros((self.N_sqrs, self.N_sqrs), dtype=np.int8)

        for idx in self.snake_body:
            grid[idx[0], idx[1]] = 1

        grid[self.food_pos[0], self.food_pos[1]] = 2

        for i in range(self.N_sqrs):
            for j in range(self.N_sqrs):
                if grid[i, j] == 1:
                    pygame.draw.rect(self.screen, self.green, pygame.Rect(i * self.dx, j * self.dy, self.dx, self.dy))
                elif grid[i, j] == 2:
                    pygame.draw.rect(self.screen, self.red, pygame.Rect(i * self.dx, j * self.dy, self.dx, self.dy))

        self.clock.tick(self.difficulty)

    def shutdown(self) -> None:
        """
        DESCRIPTION: Shutdown pygame cleanly.

        PARAMETERS: None

        RETURNS: None
        """
        if pygame is not None:
            pygame.quit()
