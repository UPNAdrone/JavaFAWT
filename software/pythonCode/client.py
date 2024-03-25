# Program: client.py
# User: Mintxoo
# Date: 01/11/2023
# Version: 2

import socket


def splitCommas(inputStr, myFans, clientNum):
    array = list(map(int, inputStr.split(",")))
    for i in range(6):
        myFans[i] = array[i + 6 * (clientNum - 1)]


def main():
    import sys

    if len(sys.argv) < 3:
        print("Specify an IP y and a puerto")
        sys.exit(-1)

    serverIp = sys.argv[1]
    serverPort = int(sys.argv[2])
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        clientSocket.connect((serverIp, serverPort))
        print("Waiting for connectivity...")

        clientNum = int(clientSocket.recv(1024).decode())
        print(f"I am client {clientNum}")

        myFans = [0] * 6
        message = "a"

        while True:
            data = clientSocket.recv(1024).decode()
            if data:
                if "Servidor dice: exit" in data:
                    print("EXIT")
                    break

                splitCommas(data, myFans, clientNum)
                print("The read fans speed are the following ones:")
                for i in range(6):
                    print(f"Fan {i}: {myFans[i]}")

                clientSocket.send(message.encode())

        clientSocket.close()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        clientSocket.close()


if __name__ == "__main__":
    main()
