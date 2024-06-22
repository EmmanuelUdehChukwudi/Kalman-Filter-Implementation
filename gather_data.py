import serial
import time

def read_from_serial(port):
    try:
        ser = serial.Serial(port, 9600, timeout=1)
        time.sleep(2) 
        while True:
            try:
                line = ser.readline()
                try:
                    line = line.decode().strip()
                except UnicodeDecodeError as e:
                    print(f"Decode error: {e}")
                    continue

                if line:
                    angles = line.split(',')
                    if len(angles) == 3:
                        x_angle = float(angles[0])
                        y_angle = float(angles[1])
                        z_angle = float(angles[2])
                        print(f"X: {x_angle}, Y: {y_angle}, Z: {z_angle}")
            except Exception as e:
                print(f"Error reading line: {e}")
                break
    except Exception as e:
        print(f"Failed to open serial port: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    # port = input("Enter the serial port (e.g., COM3 or /dev/ttyUSB0): ")
    port = "/dev/ttyUSB0"
    read_from_serial(port)
