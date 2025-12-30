from controller import Robot
from math import cos, sin
import sys
import pickle
import numpy as np

sys.path.append('../../../../controllers_shared/python_based')
from pid_controller import pid_velocity_fixed_height_controller

# ===== FLIGHT PARAMETERS =====
FLYING_ATTITUDE = 0.3
HEIGHT_STEP = 0.15
MIN_HEIGHT = 0.2
MAX_HEIGHT = 2.0

MOVE_SPEED = 0.4
YAW_SPEED = 0.05
YAW_DAMPING = 0.5

# AI BEHAVIOR TUNING
MIN_CONFIDENCE = 0.2  # Lowered to trust model more

# ACTION PERSISTENCE (reduced for more responsive behavior)
ACTION_HOLD_TIME = 0.2
YAW_HOLD_TIME = 0.4

# Model paths
MODEL_PATH = "../../../../model/models/zigzag_drone_model_v1.pkl"
SCALER_PATH = "../../../../model/models/zigzag_drone_scaler_v1.pkl"

ACTION_MAPPING = {
    0: 'hover', 1: 'forward', 2: 'backward', 3: 'left', 4: 'right',
    5: 'yaw_left', 6: 'yaw_right', 7: 'up', 8: 'down'
}

def load_ai_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    except FileNotFoundError as e:
        print(f"⚠️  Model not found: {e}")
        return None, None, False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, False

def get_ai_action_pure(model, scaler, sensor_data):
    """
    PURE model prediction - NO manipulation, NO overrides.
    """
    features = np.array([[
        sensor_data['dist_front'],
        sensor_data['dist_back'],
        sensor_data['dist_left'],
        sensor_data['dist_right'],
        sensor_data['altitude']
    ]])
    
    features_scaled = scaler.transform(features)
    
    # Direct prediction - no probability manipulation
    action = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Safety check for missing classes
    if action >= len(probabilities):
        print(f"⚠️  Model predicted invalid action {action}, using FORWARD")
        action = 1
        confidence = 0.5
    else:
        confidence = float(probabilities[action])
    
    return action, confidence, probabilities

def action_to_commands(action_id):
    """Convert action ID to movement commands."""
    forward_cmd = 0.0
    sideways_cmd = 0.0
    yaw_cmd = 0.0
    height_change = 0.0
    
    if action_id == 0:  # hover
        pass
    elif action_id == 1:  # forward
        forward_cmd = MOVE_SPEED
    elif action_id == 2:  # backward
        forward_cmd = -MOVE_SPEED
    elif action_id == 3:  # left
        sideways_cmd = MOVE_SPEED
    elif action_id == 4:  # right
        sideways_cmd = -MOVE_SPEED
    elif action_id == 5:  # yaw_left
        yaw_cmd = YAW_SPEED
    elif action_id == 6:  # yaw_right
        yaw_cmd = -YAW_SPEED
    elif action_id == 7:  # up
        height_change = HEIGHT_STEP
    elif action_id == 8:  # down
        height_change = -HEIGHT_STEP
    
    return forward_cmd, sideways_cmd, yaw_cmd, height_change

if __name__ == '__main__':
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    print("\n" + "="*70)
    print("  DRONE CONTROLLER - PURE MODEL MODE (NO OVERRIDES)")
    print("="*70)
    
    print("\n🧠 Loading AI model...")
    model, scaler, ai_available = load_ai_model()
    
    if ai_available:
        print("✅ AI model loaded successfully!")
        try:
            print(f"   Model architecture: {model.hidden_layer_sizes}")
            print(f"   Model classes: {model.classes_}")
        except Exception as e:
            print(f"   Details unavailable: {e}")
        
        # Test with known states
        print("\n🔍 Testing model predictions:")
        test_states = [
            ([1800, 1800, 1800, 1800, 0.5], "Open space"),
            ([200, 2000, 1500, 1500, 0.5], "Wall ahead"),
            ([1800, 1500, 250, 1500, 0.5], "Left wall near"),
            ([1800, 1500, 1500, 250, 0.5], "Right wall near"),
        ]
        
        for state, desc in test_states:
            test_features = np.array([state])
            test_scaled = scaler.transform(test_features)
            test_pred = model.predict(test_scaled)[0]
            test_proba = model.predict_proba(test_scaled)[0]
            print(f"   {desc:20s} -> {ACTION_MAPPING[test_pred]:10s} (conf: {test_proba[test_pred]:.2f})")
    else:
        print("⚠️  AI model not available - manual mode only")
    
    print("\n📋 Controls:")
    print("  W/S - Forward/Backward | A/D - Left/Right")
    print("  Q/E - Yaw Left/Right | R/F - Up/Down")
    if ai_available:
        print("  T - Toggle AI mode")
    print("  SPACE - Emergency hover")
    print("="*70 + "\n")

    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    motors = []
    for name in ["m1_motor", "m2_motor", "m3_motor", "m4_motor"]:
        motor = robot.getDevice(name)
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)
        motors.append(motor)
    
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)

    range_sensors = {}
    for name in ['range_front', 'range_back', 'range_left', 'range_right']:
        try:
            sensor = robot.getDevice(name)
            if sensor:
                sensor.enable(timestep)
                range_sensors[name.replace('range_', '')] = sensor
                print(f"✅ {name} enabled")
        except Exception as e:
            print(f"❌ Error enabling {name}: {e}")
    
    sensors_available = len(range_sensors) == 4

    past_x_global = 0
    past_y_global = 0
    past_time = robot.getTime()
    past_yaw = 0
    PID_CF = pid_velocity_fixed_height_controller()
    height_desired = FLYING_ATTITUDE

    ai_mode = False
    last_mode_toggle = 0
    last_print_time = 0
    print_period = 0.5
    
    current_action_id = None
    action_start_time = 0

    print("🚀 Controller ready!\n")

    try:
        while robot.step(timestep) != -1:
            now = robot.getTime()
            dt = now - past_time if (now - past_time) > 0 else 1e-6

            roll, pitch, yaw = imu.getRollPitchYaw()
            x, y, altitude = gps.getValues()
            
            yaw_rate = (yaw - past_yaw) / dt
            
            v_x_global = (x - past_x_global) / dt
            v_y_global = (y - past_y_global) / dt
            cosyaw, sinyaw = cos(yaw), sin(yaw)
            v_x = v_x_global * cosyaw + v_y_global * sinyaw
            v_y = -v_x_global * sinyaw + v_y_global * cosyaw

            dist_front = range_sensors.get('front', robot.getDevice('range_front')).getValue() if 'front' in range_sensors else 2000.0
            dist_back = range_sensors.get('back', robot.getDevice('range_back')).getValue() if 'back' in range_sensors else 2000.0
            dist_left = range_sensors.get('left', robot.getDevice('range_left')).getValue() if 'left' in range_sensors else 2000.0
            dist_right = range_sensors.get('right', robot.getDevice('range_right')).getValue() if 'right' in range_sensors else 2000.0

            sensor_data = {
                'dist_front': dist_front,
                'dist_back': dist_back,
                'dist_left': dist_left,
                'dist_right': dist_right,
                'altitude': altitude
            }

            forward_cmd = 0.0
            sideways_cmd = 0.0
            yaw_cmd = 0.0
            height_change = 0.0
            ai_confidence = 0.0
            current_action = "MANUAL"

            key = keyboard.getKey()
            yaw_input = False
            emergency_stop = False
            
            if key != -1:
                if key == ord('T') and ai_available and sensors_available:
                    if now - last_mode_toggle > 0.5:
                        ai_mode = not ai_mode
                        mode_str = "ON" if ai_mode else "OFF"
                        print(f"\n🤖 AI MODE: {mode_str}\n")
                        last_mode_toggle = now
                        current_action_id = None
                        action_start_time = 0
                
                elif key == ord(' '):
                    emergency_stop = True
                    ai_mode = False
                    current_action_id = None
                    print("\n🛑 EMERGENCY HOVER\n")
                
                elif not ai_mode:
                    if key == ord('W'):
                        forward_cmd = MOVE_SPEED
                        current_action = "MANUAL:FORWARD"
                    elif key == ord('S'):
                        forward_cmd = -MOVE_SPEED
                        current_action = "MANUAL:BACKWARD"
                    elif key == ord('A'):
                        sideways_cmd = MOVE_SPEED
                        current_action = "MANUAL:LEFT"
                    elif key == ord('D'):
                        sideways_cmd = -MOVE_SPEED
                        current_action = "MANUAL:RIGHT"
                    elif key == ord('Q'):
                        yaw_cmd = YAW_SPEED
                        yaw_input = True
                        current_action = "MANUAL:YAW_LEFT"
                    elif key == ord('E'):
                        yaw_cmd = -YAW_SPEED
                        yaw_input = True
                        current_action = "MANUAL:YAW_RIGHT"
                    elif key == ord('R'):
                        height_change = HEIGHT_STEP
                        current_action = "MANUAL:UP"
                    elif key == ord('F'):
                        height_change = -HEIGHT_STEP
                        current_action = "MANUAL:DOWN"

            if ai_mode and not emergency_stop:
                try:
                    # Get PURE model prediction - NO OVERRIDES
                    new_action_id, ai_confidence, all_probs = get_ai_action_pure(
                        model, scaler, sensor_data
                    )
                    
                    time_since_action_change = now - action_start_time
                    
                    # Action persistence for smoother behavior
                    if current_action_id is not None:
                        hold_time = YAW_HOLD_TIME if current_action_id in [5, 6] else ACTION_HOLD_TIME
                        
                        if time_since_action_change < hold_time:
                            action_id = current_action_id
                        else:
                            action_id = new_action_id
                            if action_id != current_action_id:
                                current_action_id = action_id
                                action_start_time = now
                    else:
                        action_id = new_action_id
                        current_action_id = action_id
                        action_start_time = now
                    
                    # Apply model decision directly - NO OVERRIDES
                    if ai_confidence >= MIN_CONFIDENCE:
                        forward_cmd, sideways_cmd, yaw_cmd, height_change = action_to_commands(action_id)
                        current_action = f"AI:{ACTION_MAPPING[action_id].upper()}"
                    else:
                        current_action = f"AI:LOW_CONF({ai_confidence:.2f})"
                            
                except Exception as e:
                    print(f"❌ AI prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    ai_mode = False
                    current_action_id = None
            
            # Yaw damping (only applies when not actively commanding yaw)
            if not yaw_input and abs(yaw_rate) > 0.01 and yaw_cmd == 0:
                yaw_cmd = -yaw_rate * YAW_DAMPING

            # Altitude control
            height_desired += height_change * dt
            height_desired = max(MIN_HEIGHT, min(MAX_HEIGHT, height_desired))

            # PID control
            motor_power = PID_CF.pid(dt, forward_cmd, sideways_cmd,
                                    yaw_cmd, height_desired,
                                    roll, pitch, 0,
                                    altitude, v_x, v_y)

            # Motor commands
            try:
                motors[0].setVelocity(-motor_power[0])
                motors[1].setVelocity(motor_power[1])
                motors[2].setVelocity(-motor_power[2])
                motors[3].setVelocity(motor_power[3])
            except Exception as e:
                print(f"❌ Motor command error: {e}")

            # Status logging
            if now - last_print_time > print_period:
                mode_icon = "🤖 AI" if ai_mode else "👤 MANUAL"
                status = f"{mode_icon} | t:{now:.1f}s | Alt:{altitude:.2f}m | "
                status += f"F:{dist_front:.0f} B:{dist_back:.0f} L:{dist_left:.0f} R:{dist_right:.0f} | "
                status += f"{current_action}"
                if ai_mode:
                    status += f" (conf:{ai_confidence:.2f})"
                print(status)
                last_print_time = now

            past_time = now
            past_x_global = x
            past_y_global = y
            past_yaw = yaw

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    print("\n✅ CONTROLLER STOPPED")
