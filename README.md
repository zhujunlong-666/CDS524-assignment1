# CDS524-assignment1  Enhanced Q-Learning Snake Game

## Project Overview

This project is an implementation of a reinforcement learning application using the [Q-learning](https://en.wikipedia.org/wiki/Q-learning) algorithm to train an agent to play the classic Snake game autonomously. The game takes place on a 20x20 grid where the snake learns to navigate, avoid collisions, and collect food through iterative trial-and-error interactions with the environment.

### Key Features

- **Game Environment**: A 20x20 grid (400 cells) with a snake and food items.
- **Q-Learning Algorithm**: Drives the agent's decision-making to maximize rewards over time.
- **Graphical Interface**: Built with Tkinter, featuring real-time visualization of the game state and a log section.
- **Logging**: Displays game results such as snake length, food collected, and failure reasons in the "Game Log" section.

## Introduction to Q-Learning

Q-learning is a model-free reinforcement learning technique that empowers an agent to learn optimal actions by interacting with its environment. It relies on a Q-table, which records the expected cumulative rewards for each state-action pair, refining the agent's strategy over time.

### Basic Principles

- **State**: Describes the environment's current configuration, including the snake's head position, food location, and proximity to obstacles.
- **Action**: Options available to the snake, such as moving straight, turning left, or turning right.
- **Reward**: Feedback received from the environment—positive for eating food, negative for hitting boundaries or itself.
- **Q-Table**: A lookup table mapping state-action pairs to their expected rewards.
- **Exploration vs. Exploitation**: Balances trying new actions (exploration) with leveraging known successful actions (exploitation), often via an epsilon-greedy strategy.

### Algorithm Workflow

1. Initialize the Q-table with default values.
2. Choose an action for the current state (exploring or exploiting).
3. Execute the action, observe the resulting state and reward.
4. Update the Q-table using the formula:
   ```
   Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
   ```
   where:
   - `α`: Learning rate (step size for updates).
   - `γ`: Discount factor (importance of future rewards).
   - `r`: Immediate reward.
   - `s'`: New state after the action.
5. Repeat until a termination condition (e.g., game over or target reached) is met.

## Visual Description

The game's graphical interface is a window titled "Enhanced Q-Learning Snake Game." It features:
- **Game Grid**: A 20x20 light gray grid on the left, where:
  - A **green square** represents the snake's head (e.g., positioned in the upper-middle area).
  - A **red dot** represents the food (e.g., located in the lower-middle area).
- **Log Panel**: A "Game Log" section at the bottom, displaying real-time updates (currently empty in early stages).
- **Right Panel**: A blank light gray area, potentially for future stats or visuals.

The design is minimalistic, with thin black grid lines and a clean layout, emphasizing the snake (green) and food (red) as the primary interactive elements.

## Project Structure

- **SnakeGame Class**: Handles game logic, grid rendering, and snake movement.
- **QLearningAgent Class**: Implements the Q-learning algorithm to guide the snake.
- **main Function**: Orchestrates the game loop, initializes components, and manages user input.

## How to Run the Project

### Requirements

- Python 3.x
- Tkinter (typically bundled with Python)
- NumPy (`pip install numpy`)

### Steps to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install numpy
   ```

3. **Launch the Game**:
   ```bash
   python main.py
   ```

4. **Set Winning Condition**:
   - A dialog box will prompt you to enter a target snake length (e.g., 15 or 30).

5. **Observe Gameplay**:
   - The snake navigates autonomously using Q-learning.
   - The "Game Log" updates with session results (e.g., length achieved, food eaten).

## Contributing

We welcome contributions! Please submit issues or pull requests to enhance the project.

## License

This project is licensed under the [MIT License](LICENSE).
