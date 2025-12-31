from controller import Robot
from math import cos, sin
import sys
import pickle
import numpy as np

sys.path.append('../../../../controllers_shared/python_based')
from pid_controller import pid_velocity_fixed_height_controller

FLYING_ATTITUDE = 0.3
HEIGHT_STEP = 0.15
MIN_HEIGHT = 0
MAX_HEIGHT = 2.0

MOVE_SPEED = 0.4
YAW_SPEED = 0.05
YAW_DAMPING = 0.5

ACTION_HOLD_TIME = 0.2
YAW_HOLD_TIME = 0.4

MODEL_PATH = "../../../../model/models/v1_zigzag/drone_mlp_model.pkl"
SCALER_PATH = "../../../../model/models/v1_zigzag/drone_scaler.pkl"

ACTION_MAPPING = {
    0: 'hover', 1: 'forward', 2: 'backward', 3: 'left', 4: 'right',
    5: 'yaw_left', 6: 'yaw_right', 7: 'up', 8: 'down'
}

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Model loaded successfully!")
        try:
            print(f"   Architecture: {model.hidden_layer_sizes}")
            print(f"   Features: 6 (F/B/L/R/Down/Up)")
        except:
            pass
        return model, scaler, True
    except FileNotFoundError as e:
        print(f"⚠️  Model not found: {e}")
        return None, None, False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, False


def get_ai_action(model, scaler, sensor_data):
    features = np.array([[
        sensor_data['dist_front'],
        sensor_data['dist_back'],
        sensor_data['dist_left'],
        sensor_data['dist_right'],
        sensor_data['dist_down'],
        sensor_data['dist_up']
    ]])

    features_scaled = scaler.transform(features)
    action = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]

    if action >= len(probabilities):
        print(f"⚠️  Invalid action {action}, using FORWARD")
        action = 1
        confidence = 0.5
    else:
        confidence = float(probabilities[action])

    return action, confidence


def action_to_commands(action_id):
    forward_cmd = 0.0
    sideways_cmd = 0.0
    yaw_cmd = 0.0
    height_change = 0.0

    if action_id == 0:
        pass
    elif action_id == 1:
        forward_cmd = MOVE_SPEED
    elif action_id == 2:
        forward_cmd = -MOVE_SPEED
    elif action_id == 3:
        sideways_cmd = MOVE_SPEED
    elif action_id == 4:
        sideways_cmd = -MOVE_SPEED
    elif action_id == 5:
        yaw_cmd = YAW_SPEED
    elif action_id == 6:
        yaw_cmd = -YAW_SPEED
    elif action_id == 7:
        height_change = HEIGHT_STEP
    elif action_id == 8:
        height_change = -HEIGHT_STEP

    return forward_cmd, sideways_cmd, yaw_cmd, height_change


if __name__ == '__main__':
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    print("\n" + "=" * 70)
    print("  DRONE AI CONTROLLER - 6-FEATURE MODEL (F/B/L/R/Down/Up)")
    print("=" * 70)

    model, scaler, success = load_model()
    if not success:
        print("\n❌ Model failed to load. Exiting.")
        sys.exit(1)

    print("\n📋 Controls:")
    print("  W/S - Forward/Backward | A/D - Left/Right")
    print("  Q/E - Yaw Left/Right   | R/F - Up/Down")
    print("  T - Toggle AI mode ON/OFF")
    print("  SPACE - Emergency hover")
    print("=" * 70 + "\n")

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
    for name in ['range_front', 'range_back', 'range_left', 'range_right', 'range_down', 'range_up']:
        try:
            sensor = robot.getDevice(name)
            if sensor:
                sensor.enable(timestep)
                range_sensors[name.replace('range_', '')] = sensor
                print(f"✅ Enabled {name}")
        except Exception as e:
            print(f"❌ Error enabling {name}: {e}")

    sensors_available = all(k in range_sensors for k in ['front', 'back', 'left', 'right', 'down', 'up'])
    
    if not sensors_available:
        print("\n⚠️  WARNING: Not all required sensors found!")
        print(f"    Available: {list(range_sensors.keys())}")

    past_x_global = 0.0
    past_y_global = 0.0
    past_time = robot.getTime()
    past_yaw = 0.0
    PID_CF = pid_velocity_fixed_height_controller()
    height_desired = FLYING_ATTITUDE

    ai_mode = False
    last_mode_toggle = 0.0
    last_print_time = 0.0
    print_period = 0.5

    current_action_id = None
    action_start_time = 0.0

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

            dist_front = range_sensors.get('front').getValue() if 'front' in range_sensors else 2000.0
            dist_back = range_sensors.get('back').getValue() if 'back' in range_sensors else 2000.0
            dist_left = range_sensors.get('left').getValue() if 'left' in range_sensors else 2000.0
            dist_right = range_sensors.get('right').getValue() if 'right' in range_sensors else 2000.0
            dist_down = range_sensors.get('down').getValue() if 'down' in range_sensors else 2000.0
            dist_up = range_sensors.get('up').getValue() if 'up' in range_sensors else 2000.0

            sensor_data = {
                'dist_front': dist_front,
                'dist_back': dist_back,
                'dist_left': dist_left,
                'dist_right': dist_right,
                'dist_down': dist_down,
                'dist_up': dist_up
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
                if key == ord('T') and sensors_available:
                    if now - last_mode_toggle > 0.5:
                        ai_mode = not ai_mode
                        mode_str = "ON" if ai_mode else "OFF"
                        print(f"\n🤖 AI MODE: {mode_str}\n")
                        last_mode_toggle = now
                        current_action_id = None
                        action_start_time = 0.0

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
                    new_action_id, ai_confidence = get_ai_action(model, scaler, sensor_data)

                    time_since_action_change = now - action_start_time

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

                    forward_cmd, sideways_cmd, yaw_cmd, height_change = action_to_commands(action_id)
                    current_action = f"AI:{ACTION_MAPPING[action_id].upper()}"

                except Exception as e:
                    print(f"❌ AI prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    ai_mode = False
                    current_action_id = None

            if not yaw_input and abs(yaw_rate) > 0.01 and yaw_cmd == 0.0:
                yaw_cmd = -yaw_rate * YAW_DAMPING

            height_desired += height_change * dt
            height_desired = max(MIN_HEIGHT, min(MAX_HEIGHT, height_desired))

            motor_power = PID_CF.pid(
                dt, forward_cmd, sideways_cmd,
                yaw_cmd, height_desired,
                roll, pitch, 0.0,
                altitude, v_x, v_y
            )

            try:
                motors[0].setVelocity(-motor_power[0])
                motors[1].setVelocity(motor_power[1])
                motors[2].setVelocity(-motor_power[2])
                motors[3].setVelocity(motor_power[3])
            except Exception as e:
                print(f"❌ Motor command error: {e}")

            if now - last_print_time > print_period:
                mode_icon = "🤖" if ai_mode else "👤"
                status = f"{mode_icon} | t:{now:.1f}s | Alt:{altitude:.2f}m | "
                status += f"F:{dist_front:.0f} B:{dist_back:.0f} L:{dist_left:.0f} R:{dist_right:.0f} | "
                status += f"D:{dist_down:.0f} U:{dist_up:.0f} | {current_action}"
                
                if ai_mode:
                    status += f" (c:{ai_confidence:.2f})"

                print(status)
                last_print_time = now

            past_time = now
            past_x_global = x
            past_y_global = y
            past_yaw = yaw

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    print("\n✅ CONTROLLER STOPPED")
