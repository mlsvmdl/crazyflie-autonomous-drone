# Crazyflie Autonomous Drone Simulation

This project simulates a Crazyflie-style quadcopter in Webots, using a neural network trained on synthetic expert data for autonomous obstacle avoidance. The drone is equipped with six distance sensors (front, back, left, right, down, up) and uses a trained multi-layer perceptron (MLP) classifier to map sensor readings to one of nine discrete actions, enabling fully autonomous navigation in complex environments.

While this project also supports Gazebo simulation, this README focuses on the Webots-based AI controller workflow. For general simulator installation and manual control modes, see the documentation in `docs/`.

## Repository Structure

```
.
├── controllers_shared/          # Shared PID controller implementations (C and Python)
├── docs/                         # Installation guides and user documentation
│   ├── installing/              # Webots and Gazebo installation instructions
│   └── user_guides/             # Keyboard control, wall following, etc.
├── meshes/                       # 3D assets (Blender, Collada, STL files, textures)
├── model/                        # Machine learning pipeline (data generation, training, models)
│   ├── data/                    # Synthetic data generation scripts
│   ├── models/                  # Trained model artifacts
│   └── drone_env/               # Python virtual environment (created by user)
├── simulator_files/
│   ├── gazebo/                  # Gazebo-specific files
│   └── webots/                  # Webots controllers, protos, and world files
└── README.md                     # This file
```

## Environment Setup

Before generating data or training the model, set up a Python virtual environment:

```bash
# Navigate to the model directory
cd model

# Create virtual environment
python3 -m venv drone_env

# Activate the environment (Linux/macOS)
source drone_env/bin/activate

# On Windows, use:
# drone_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The virtual environment will be created at `model/drone_env/` and should be activated whenever you work with the ML components.

## Data Generation

The script `model/data/generate_data_zigzag.py` generates synthetic training data for the drone by encoding expert navigation rules.

### How It Works

- **Inputs**: Six distance sensor readings (front, back, left, right, down, up)
- **State Classification**: Distances are classified into states based on thresholds:
  - Critical (very close)
  - Near
  - Close
  - Approaching
  - Safe
- **Expert Rules**: Based on sensor states, the script assigns one of 9 actions:
  - `0`: Hover (stay in place)
  - `1`: Forward
  - `2`: Backward
  - `3`: Left
  - `4`: Right
  - `5`: Yaw left
  - `6`: Yaw right
  - `7`: Up
  - `8`: Down
- **Dataset Balancing**: Ensures each action has sufficient samples for training

### Running Data Generation

From the repository root:

```bash
cd model
python data/generate_data_zigzag.py
```

This outputs the training dataset to `model/zigzag_drone_training_data.csv`.

## Model Training

The script `model/zigzag_fast_train.py` trains a neural network classifier on the generated data.

### Training Pipeline

1. Loads `zigzag_drone_training_data.csv`
2. Splits data into training and test sets
3. Applies `StandardScaler` to normalize the six sensor features
4. Trains a scikit-learn `MLPClassifier` with:
   - Hidden layers: `(128, 64, 32)`
   - Activation: ReLU
   - Optimizer: Adam
   - Early stopping enabled
5. Evaluates performance (accuracy, per-class metrics)
6. Saves trained artifacts to `model/models/v1_zigzag/`:
   - `drone_mlp_model.pkl` (trained classifier)
   - `drone_scaler.pkl` (feature scaler)

### Running Training

From the repository root:

```bash
cd model
python zigzag_fast_train.py
```

Training metrics will be printed to the console, and model files will be saved for use in simulation.

## Webots AI Controller

The trained model is deployed in Webots through a Python controller located in `simulator_files/webots/controllers/`.

### Webots File Structure

```
simulator_files/webots/
├── controllers/        # Python controller scripts (including AI controller)
├── protos/            # Robot and object prototypes
└── worlds/            # Simulation world files
```

### Launching the Simulation

1. **Start Webots** and open a world file:

```bash
webots simulator_files/webots/worlds/<your_world>.wbt
```

2. **Configure the Robot Controller**:
   - In the Webots Scene Tree, select the Crazyflie robot node
   - Set the `controller` field to `"crazyflie_mlp_controller"` (or your AI controller name)

3. **Verify Controller Configuration**:

The AI controller script (in `simulator_files/webots/controllers/`) should have:

```python
# Model paths (relative to controller directory)
MODEL_PATH = "../../../model/models/v1_zigzag/drone_mlp_model.pkl"
SCALER_PATH = "../../../model/models/v1_zigzag/drone_scaler.pkl"

# Enable AI mode
USE_AI_MODE = True
```

### How the AI Controller Works

Each simulation step, the controller:

1. **Reads sensor data**: Queries the six distance sensors on the drone
2. **Builds feature vector**: Constructs input in the order expected by the model
3. **Normalizes features**: Applies the saved `StandardScaler`
4. **Predicts action**: Uses the trained MLP to classify the current state into one of 9 actions
5. **Optional smoothing**: May average actions over multiple timesteps for stability
6. **Executes command**: Sends motor velocities through the PID controller (from `controllers_shared/python_based/pid_controller.py`)

The drone will autonomously navigate, avoiding obstacles based on its trained behavior.

## End-to-End Usage

Complete workflow from setup to simulation:

```bash
# 1. Set up environment
cd model
python3 -m venv drone_env
source drone_env/bin/activate
pip install -r requirements.txt

# 2. Generate training data
python data/generate_data_zigzag.py

# 3. Train the model
python zigzag_fast_train.py

# 4. Launch Webots (from repository root)
cd ..
webots simulator_files/webots/worlds/<your_world>.wbt

# 5. In Webots GUI:
#    - Select the Crazyflie robot
#    - Set controller to "crazyflie_mlp_controller"
#    - Verify USE_AI_MODE = True in controller code
#    - Run simulation and observe autonomous navigation
```

## Other Documentation

For additional information:

- **Simulator Installation**: See `docs/installing/install_webots.md` and `docs/installing/install_gazebo.md`
- **Manual Control**: See `docs/user_guides/webots_keyboard_control.md`
- **Rule-Based Control**: See `docs/user_guides/webots_wall_following.md`

This README focuses specifically on the neural network-based AI controller. The documentation folder contains guides for other control modes and simulator configurations.

## License

This project is licensed under the terms in the `LICENSE` file.

## Contributions

Contributions are welcome! Please feel free to submit issues or pull requests to improve the project.
