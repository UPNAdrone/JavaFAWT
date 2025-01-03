# Program: pwm_client.py
# User: Mintxoo
# Date: 05/12/2023
# Version: 1

import socket
import RPi.GPIO as GPIO


def splitCommas(inputStr, myFans, clientNum):
    array = list(map(int, inputStr.split(",")))
    print()
    for i in range(6):
        speed = array[i + 6 * (clientNum - 1)]
        myFans[i].ChangeDutyCycle(speed)
        print(i,": ",speed)	


def main():
    import sys

    GPIO.setmode(GPIO.BOARD)

    if len(sys.argv) < 3:
        print("Specify an IP and a port")
        sys.exit(-1)

    server_ip = sys.argv[1]
    server_port = int(sys.argv[2])
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        client_socket.connect((server_ip, server_port))
        print("Waiting for connectivity...")

        client_number = int(client_socket.recv(1024).decode())
        print(f"I am client {client_number}")

        my_fans = [0] * 6
        message = "a"

        GPIO.setup(37, GPIO.OUT)
        my_fans[0] = GPIO.PWM(37, 5000)
        GPIO.setup(35, GPIO.OUT)
        my_fans[1] = GPIO.PWM(35, 5000)
        GPIO.setup(33, GPIO.OUT)
        my_fans[2] = GPIO.PWM(33, 5000)
        GPIO.setup(31, GPIO.OUT)
        my_fans[3] = GPIO.PWM(31, 5000)
        GPIO.setup(29, GPIO.OUT)
        my_fans[4] = GPIO.PWM(29, 5000)
        GPIO.setup(23, GPIO.OUT)
        my_fans[5] = GPIO.PWM(23, 5000)

        for i in range(6):
            my_fans[i].start(0)

        while True:
            data = client_socket.recv(1024).decode()
            if data:
                if "Server says: exit" in data:
                    print("EXIT")
                    break

                splitCommas(data, my_fans, client_number)
                client_socket.send(message.encode())

        client_socket.close()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()


if __name__ == "__main__":
    main()
