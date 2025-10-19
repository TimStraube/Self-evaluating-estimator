import json
import numpy as np
import random
import os

class ARCEnvironment:
    """
    ARC (Abstraction and Reasoning Corpus) Environment for SEE network.
    Presents ARC tasks as reinforcement learning problems.
    """

    def __init__(self, data_path=None, task_ids=None, max_grid_size=30):
        # Load ARC data
        self.data_path = data_path or "/home/timstraube/Programme/SEE/data/arc-prize-2025"
        self.tasks = {}
        self.task_ids = task_ids or []
        self.current_task_id = None
        self.current_example_idx = 0
        self.max_grid_size = max_grid_size

        # Load training data
        self.load_arc_data()

        # Current state
        self.current_input = None
        self.current_output = None
        self.current_prediction = None
        self.step_count = 0
        self.max_steps = 50  # Maximum steps per episode

    def load_arc_data(self):
        """Load ARC training data"""
        train_file = os.path.join(self.data_path, "arc-agi_training_challenges.json")

        try:
            with open(train_file, 'r') as f:
                self.tasks = json.load(f)
                print(f"Loaded {len(self.tasks)} ARC tasks")

                if not self.task_ids:
                    self.task_ids = list(self.tasks.keys())[:10]  # Use first 10 tasks for now

        except FileNotFoundError:
            print(f"ARC training file not found: {train_file}")
            # Create dummy task for testing
            self.tasks = self._create_dummy_task()

    def _create_dummy_task(self):
        """Create a simple dummy task for testing"""
        return {
            "dummy_task": {
                "train": [
                    {
                        "input": [[1, 0], [0, 1]],
                        "output": [[1, 1], [1, 1]]
                    }
                ],
                "test": [
                    {
                        "input": [[0, 1], [1, 0]]
                    }
                ]
            }
        }

    def reset(self, seed=None, options=None):
        """Reset environment to start of new episode"""
        # seed und options werden aktuell nicht verwendet, aber für Gymnasium-Kompatibilität akzeptiert
        # Select random task
        if self.task_ids:
            self.current_task_id = random.choice(self.task_ids)
        else:
            self.current_task_id = list(self.tasks.keys())[0]

        task = self.tasks[self.current_task_id]

        # Select random training example
        if 'train' in task and task['train']:
            example = random.choice(task['train'])
            self.current_input = np.array(example['input'])
            self.current_output = np.array(example['output'])
        else:
            # Fallback to dummy
            self.current_input = np.array([[1, 0], [0, 1]])
            self.current_output = np.array([[1, 1], [1, 1]])

        # Initialize prediction as zeros with output shape
        self.current_prediction = np.zeros_like(self.current_output)
        self.step_count = 0

        # Create observation with input and prediction channels
        input_padded = self._pad_grid(self.current_input)
        prediction_padded = self._pad_grid(self.current_prediction)
        observation = np.stack([input_padded, prediction_padded], axis=0)

        return observation

    def step(self, action):
        """Execute one step in the environment"""
        self.step_count += 1

        # Action format: [memory_slot, grid_action]
        memory_slot, grid_action = action

        # Convert grid_action to grid coordinates
        grid_height, grid_width = self.current_prediction.shape
        row = grid_action // grid_width
        col = grid_action % grid_width

        # Simple action: toggle the cell value (0->1, 1->0, etc.)
        if 0 <= row < grid_height and 0 <= col < grid_width:
            # Cycle through possible values (0-9 for ARC)
            current_value = self.current_prediction[row, col]
            self.current_prediction[row, col] = (current_value + 1) % 10

        # Reward wird im Agenten berechnet, hier nur externer Reward (z.B. 0)
        reward = 0.0

        # Check if episode is done
        done = (self.step_count >= self.max_steps or
                self._is_correct_solution())

        # Next observation
        input_padded = self._pad_grid(self.current_input)
        prediction_padded = self._pad_grid(self.current_prediction)
        observation = np.stack([input_padded, prediction_padded], axis=0)

        return observation, reward, done, {}

    def _pad_grid(self, grid):
        """Pad grid to maximum size"""
        if grid is None:
            return np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32)

        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32)
        h, w = grid.shape
        padded[:h, :w] = grid
        return padded

    def _compute_reward(self):
        """SEE-Prinzip: Reward = negative Summe der absoluten Differenzen zwischen Vorhersage und Ziel"""
        if self.current_prediction is None or self.current_output is None:
            return -1.0

        # Wahrnehmungsdissonanz
        delta_o = self.current_prediction - self.current_output
        r_delta = -np.sum(np.abs(delta_o))
        # Optional: kleine Schrittstrafe
        reward = r_delta - 0.01
        return float(reward)

    def _is_correct_solution(self):
        """Check if current prediction matches the target output"""
        if self.current_prediction is None or self.current_output is None:
            return False

        return np.array_equal(self.current_prediction, self.current_output)

    def render(self):
        """Render current state"""
        print(f"Task: {self.current_task_id}")
        print("Input:")
        print(self.current_input)
        print("Current Prediction:")
        print(self.current_prediction)
        print("Target Output:")
        print(self.current_output)
        print(f"Step: {self.step_count}/{self.max_steps}")

    def get_task_info(self):
        """Get information about current task"""
        return {
            'task_id': self.current_task_id,
            'input_shape': self.current_input.shape if self.current_input is not None else None,
            'output_shape': self.current_output.shape if self.current_output is not None else None,
            'step_count': self.step_count
        }