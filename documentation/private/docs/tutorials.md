# Tutorial / Manual of use



The following tutorial is just a guidance to the use of the software. If there is something that is not understood in the tutorial, it is recommended to visit the software documentation. There you will find much more information about each class.


In case you need to change some connection protocol, ip address, port number,... just open the software documentation and there you will find where to change it.


In any other case, please contact us.


## INTRODUCTION


To control the fan modules there is needed a computer and some raspberries. The raspberries are the ones that modify the speed of the fans, but we want to do these modifications from our computer. Hence, the computer has to be connected to each one of the raspberries.



## CLIENT


### Raspberry pi settings (repeat for each raspberry)
1.  The clients that will be physically connected to the fan modules are raspberries. You can use either raspberries pi 0, 3 or 4. In my case I have tried with the three of them. To configure them you can use the “Raspberry Pi Imager” app, following the instructions of this document, https://bricolabs.cc/wiki/guias/guiadeinicioraspberrypi2021 . Make sure you put a different hostname for every raspberry, so that later you can distinguish them.


2.  After that, you will connect your raspberry to the current. Now, open the “Advanced IP Scanner” app to scan all the active ip addresses on your net. If the configuration has been done correctly you will see your raspberries with their hostnames and their ip addresses.


3.  Now, using the “PuTTY” app, connect to your raspberries introducing their ip address. When you do that, a raspberry terminal window will open. There you can navigate through its files and folders.


4.  Go to “/software/clientCode” folder and there you will find a single file called pwm_client. This file is used for the raspberry to connect to the server.


5.  On the raspberry’s terminal, find a location to save this file (for example on /desktop). Once you found it, type “nano pwm_client.py” and paste the content of the pwm_client.py in this file. Make sure you know the location of the file. Repeat it in all the raspberries.

6. To execute the file you need some previous installations (type the following on this directory):
    * sudo apt install python3
    * pip3 install RPi.GPIO

7. Your raspberries are ready to work. Do not forget that it requires 5V to work properly. More volts can break it.


### Connect fans to Raspberry (repeat for each raspberry)
1.  Mount the design for the connection of the cables on the raspberry. It is put on its pins.


2. There are two places to connect current of 12V and 6V.


3. Then you will find 6 different modules. Each of them corresponds to a fan. Each fan has cables with different colors: PWM (blue), TACH (yellow), VIN (red) and GND (black) to be connected. This link can help identify where each cable goes (https://mans.io/files/viewer/1709981/1 ) by looking at the colors. The first fan is the one closer to the 12V current. Then connect them in order.




### Connection (if the Raspberry pi settings has been already done)
1.  Connect the raspberries to the current. 


2.  Open Advanced IP Scanner to look for their IP addresses and use PuTTY to connect to each one of them.


3.  In the terminals, search for the location of pwm_client.py. To execute it you need to introduce the IP address and a port number of the server. So, before executing the file, make sure the server is active and listening for clients.


4.  Next, once in the folder of pwm_client.py, type the following line to execute the client: python pwm_client.py “server_ip” “port”. (for example: python pwm_client.py 192.168.1.33 125)


5.  If the connection is successful, it will be printed, “I am client x”, being x the number of clients assigned to you. You are now connected to the server.


6.  From now on, you won’t have to use the raspberries anymore. During the execution, the raspberries will be printing the messages they are receiving in the terminal, just if necessary.


##  SERVER
1.  Enter the “/software/serverCode/target” folder.


2.  Execute the .jar file called "serverCode-1.0-SNAPSHOT.jar”. You may need to do it through the terminal.


3.  A window called FAN CONNECTION will open. There you will see your server ip and your server port (comment1). Below, is a list where the connected clients will show up. When a client successfully connects to the server, it will appear on the list.
Once you see all the clients connected, you can click on the Ready button to continue.


4.  Two new windows will open. The window called SPEED SCHEMA shows the changing speed of the different active fans. It works with red colored circles. When a specific fan increases its spinning speed, the circle in its position will become bigger. The same happens with all the other positions of the fans. This window is just a visual representation of the speeds of all the active fans.


5.  Moreover, the FAN CONTROL window is the one used to control the fans. There are two different ways to control them.
    - Manually: At the left side of the window, you can see the Fans schema, where you will be able to select your active fans (comment2). You can select several fans by clicking on them. Once you select one of them, on the upper right part, you will see its current speed. The selected fans will be in green color, while the other ones are in white (the inactive fans are in red).
Use the scrolling bar to set the speed you want for the selected fans. Once set, click on the Update button to send the speed to the clients. The selected fans will automatically start spinning at that speed and on the SPEED SCHEMA you will see a change in its circle dimensions (depending on the new speed). To stop all the fans, click on the Stop All button.
    - By a functionality file: If you don’t want to manually set every fan speed you can also use a functionality file. The file must be in format .xlsx to be read correctly. The first column is for the time and the next columns are for the fan's speed. You have some examples in the “/software/functionalities” folder. To use it, click on the Add Functionality Button on the window. Drag and drop the file. Once added you will see its name below. Now you have two options:
        - Real execution: If you click on the Execute Functionality button, the fans will receive their new speeds and the functionality will execute in real time. To stop it you can click on Stop Execution, but notice that this button just stops the reading of the files, the fans will keep spinning at the last speed they received.. During this execution, the manual fan control will be disabled.
        - Preview execution: If you click on Execute Preview the fans will not change. A window like SPEED SCHEMA will open, and there you will be able to see what will happen to the fans if you execute the functionality. To close it, just close the window.


6. Apart from this, you can also click on Show Pressure to see the pressure values in real time on a window. For this, you will need a specific pressure sensor. The first time you open it you will have to introduce its IP address and its port number. In our case we have used a nanodaq-lts-32, that has 32 reading points.


7. Before closing the program don’t forget to Stop All the fans. Otherwise, they will keep the last speed value you gave them, and as soon as they connect to current, they will spin at this speed.


##  FUNCTIONALITY FILES


1. In the software folder, inside functionalities, there are some example files that can be used to execute the fans with them.


2. In case you do not want to use them, you can create new ones or just modify the existing ones. But remember that they need to have the .svg extension.


3. The files are read row by row. Each row represents a speed update. The first column is the time (in milliseconds) you want the fans to be spinning at these speeds, while the other columns are the speeds of each fan (in percentage), in order.
 
### Comments
- Comment 1
The default server port is the 125. In case you want to change it, go to the serverCode project, in the folder “x”. Open the class “Main.java”. In the line 14, replace “125” with the port you wish.


- Comment 2
Once pressed the ready button. The software will check how many clients are connected to the server to display their fans. If only one client is connected, the only available fans will be from 1 to 6. If there are two, from 1 to 12, and so on. The available fans appear in white color and can be clicked, while the not available ones appear in red color and cannot be clicked.
