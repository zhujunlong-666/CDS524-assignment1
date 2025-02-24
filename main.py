import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
import time
import random
import numpy as np

class SnakeGame:
    def __init__(self, master, grid_size=20, cell_size=25):
        self.master = master
        self.grid_size = grid_size
        self.cell_size = cell_size

        self.canvas = tk.Canvas(master, width=grid_size * cell_size,
                               height=grid_size * cell_size, bg='lightgray')
        self.canvas.pack(expand=True, fill='both', padx=50, pady=20)
        
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.reset_game()
        
        self.info_label = tk.Label(master, text="", font=('Arial', 12))
        self.info_label.pack(pady=10)
        self.log_area = scrolledtext.ScrolledText(master, width=80, height=10,
                                                 font=('Consolas', 10))
        self.log_area.pack(pady=5)
        self.log_area.insert(tk.END, "Game Log:\n")
        self.log_area.configure(state='disabled')
        
    def reset_game(self):
        self.snake = [(10, 10)]
        self.direction = random.choice([0, 1, 2, 3])
        self.food = self.generate_food()
        self.dead = False
        self.steps = 0
        self.score = 1
        self.food_eaten = 0
        self.death_reason = ""
        
    def generate_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food
    
    def calculate_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_possible_actions(self):
        forbidden = (self.direction + 2) % 4
        return [a for a in [0, 1, 2] if (self.direction + a - 1) % 4 != forbidden]

    def move(self, action):
        valid_actions = self.get_possible_actions()
        if action not in valid_actions:
            action = 0  # 默认直行
            
        new_dir = (self.direction + (action - 1)) % 4
        self.direction = new_dir
        
        old_head = self.snake[0]
        old_distance = self.calculate_distance(old_head, self.food)
        
        dx, dy = self.directions[self.direction]
        new_head = (
            (old_head[0] + dx) % self.grid_size,
            (old_head[1] + dy) % self.grid_size
        )
        
        reward = 0.1  # 生存奖励
        new_distance = self.calculate_distance(new_head, self.food)
        
        if new_head in self.snake:
            self.dead = True
            self.death_reason = "Hit itself"
            reward = -10
        elif new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self.generate_food()
            self.score += 1
            self.food_eaten += 1
            reward = 10 + (self.score / 10)
        else:
            self.snake = [new_head] + self.snake[:-1]
            
        reward += (old_distance - new_distance) * 0.5
        self.steps += 1
        return reward, self.dead
    
    def draw(self):
        self.canvas.delete("all")
        for i in range(self.grid_size + 1):
            self.canvas.create_line(0, i * self.cell_size,
                                   self.grid_size * self.cell_size, i * self.cell_size,
                                   fill='gray', width=1)
            self.canvas.create_line(i * self.cell_size, 0,
                                   i * self.cell_size, self.grid_size * self.cell_size,
                                   fill='gray', width=1)
        
        for idx, (x, y) in enumerate(self.snake):
            color = 'black' if idx == 0 else 'gray'
            self.canvas.create_rectangle(
                x * self.cell_size, y * self.cell_size,
                (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                fill=color, outline='white'
            )
        
        fx, fy = self.food
        self.canvas.create_oval(
            fx * self.cell_size, fy * self.cell_size,
            (fx + 1) * self.cell_size, (fy + 1) * self.cell_size,
            fill='red', outline='white'
        )

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.2, epsilon_decay=0.999, min_epsilon=0.01):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
    def get_state_key(self, game):
        head = game.snake[0]
        food = game.food
        
        dx = food[0] - head[0]
        dy = food[1] - head[1]
        fx_dir = 0 if dx == 0 else 1 if dx > 0 else -1
        fy_dir = 0 if dy == 0 else 1 if dy > 0 else -1
        
        danger = []
        for d in [game.direction, (game.direction + 1) % 4, (game.direction - 1) % 4]:
            move = game.directions[d]
            next_pos = (
                (head[0] + move[0]) % game.grid_size,
                (head[1] + move[1]) % game.grid_size
            )
            danger.append(1 if next_pos in game.snake else 0)
            
        distance = game.calculate_distance(head, food)
        distance_feature = min(distance // 5, 4)
        length_feature = min(len(game.snake) // 5, 6)
        
        return (fx_dir, fy_dir, tuple(danger), distance_feature, game.direction, length_feature)
    
    def choose_action(self, game, state):
        if random.random() < self.epsilon:
            return random.choice(game.get_possible_actions())
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
            
        valid_actions = game.get_possible_actions()
        q_values = [self.q_table[state][a] if a in valid_actions else -np.inf
                   for a in range(3)]
        
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        
        if len(best_actions) > 1:
            action_distances = []
            for a in best_actions:
                new_dir = (game.direction + (a - 1)) % 4
                dx, dy = game.directions[new_dir]
                new_head = (
                    (game.snake[0][0] + dx) % game.grid_size,
                    (game.snake[0][1] + dy) % game.grid_size
                )
                action_distances.append(game.calculate_distance(new_head, game.food))
            return best_actions[np.argmin(action_distances)]
        
        return np.argmax(q_values)
    
    def update_q_table(self, state, action, reward, new_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        if new_state not in self.q_table:
            self.q_table[new_state] = np.zeros(3)
            
        target = reward + self.gamma * np.max(self.q_table[new_state])
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])
        
    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def main():
    root = tk.Tk()
    root.title("Optimized Q-learning Snake Game")
    window_width, window_height = 800, 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.resizable(False, False)
    
    game = SnakeGame(root)
    game.draw()
    root.update()
    
    win_length = simpledialog.askinteger("Input", "Please enter the winning length:",
                                        parent=root)
    if not win_length:
        return
    
    agent = QLearningAgent(epsilon=0.2, epsilon_decay=0.999, min_epsilon=0.01)
    total_games = 0
    start_time = time.time()
    
    def add_log_entry():
        game.log_area.configure(state='normal')
        log_text = f"Game {total_games} | Length: {game.score} | " \
                   f"Food Eaten: {game.food_eaten} | "
        if game.dead:
            log_text += f"Failed: {game.death_reason}"
        elif game.score >= win_length:
            log_text += "Game Won"
        game.log_area.insert(tk.END, log_text + "\n")
        game.log_area.see(tk.END)
        game.log_area.configure(state='disabled')
    
    def game_loop():
        nonlocal total_games
        game.reset_game()
        total_games += 1
        state = agent.get_state_key(game)
        
        while not game.dead:
            action = agent.choose_action(game, state)
            reward, done = game.move(action)
            new_state = agent.get_state_key(game)
            agent.update_q_table(state, action, reward, new_state)
            
            state = new_state
            game.draw()
            elapsed = time.time() - start_time
            game.info_label.config(text=f"Games Played: {total_games} | "
                                      f"Current Length: {game.score} | "
                                      f"Steps: {game.steps} | Total Time: {elapsed:.1f}s")
            root.update()
            
            if game.score >= win_length:
                add_log_entry()
                messagebox.showinfo("Victory", f"Reached target length {win_length}!")
                root.destroy()
                return
            
            if done:
                add_log_entry()
                agent.decay_epsilon()
                root.after(50, game_loop)
                return
            
        root.after(50, game_loop)
    
    game_loop()
    root.mainloop()

if __name__ == "__main__":
    main()