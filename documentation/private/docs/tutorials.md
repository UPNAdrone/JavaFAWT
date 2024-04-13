# Tutorial / Manual of use

## CLIENT

### Raspberries settings
1.	The clients that will be physically connected to the fan modules are raspberries. You can use either raspberries pi 3 or 4. In my case I use both. To configure them you can use Raspberry Pi Imager app, following the instructions of this document, https://bricolabs.cc/wiki/guias/guiadeinicioraspberrypi2021 . Make sure you put a different hostname for every raspberry, so that later you can distinguish them.

2.	After that, you will connect your raspberries to current. Now, open the Advanced IP Scanner app to scan all the active ip addresses on your net. If the configuration has been done correctly you will see your raspberries and their ip addresses. 

3.	Now, using the PuTTY app, connect to each one of your raspberries introducing their ip address. When you do that, for every one of your raspberries a new terminal window will open. There you can navigate through its files and folders. 

4.	Open the pythonCode folder (for example on Visual Studio). The file used for the raspberries to connect to the server, is the one called pwm_client.py. 

5.	On the raspberry’s terminal, find a place to save this file. Once you find the place you can do nano pwm_client.py and in this file paste the content of the pwm_client.py file. Make sure you know the location of the file.

### Connection (if the previous configuration has been already done)
1.	Connect the raspberries to current.

2.	Open Advanced IP Scanner and with PuTTY connect to each one of them.

3.	In the terminals, search for the location of pwm_client.py. To execute it you need to introduce the IP address and a port number of the server. So, before executing the file, make sure the server is active and listening for clients.

4.	Next, execute the file like this: python pwm_client.py “ip” “port”.

5.	If the connection is successful, it will be printed, I am client x. You are now connected to the server.

6.	From now on, you won’t have to use the raspberries anymore. 

##  SERVER
1.	Enter the software folder.

2.	Open the serverCode project on NetBeans.

3.	Run the project, and it will by default execute the Main.java class. 

4.	A window called FAN CONECTION will open. There you will see your server ip and your server port (comment1). Below, is a list where the connected clients will show up. Once you see all the clients connected, you can click on the Ready button to continue.

5.	To connect the clients to the server, you will have to open the clientDocumentation file in the documentation folder.

6.	If the client successfully connects to the server, on the terminal a similar message will be displayed: I am client “x”. And on the server client list, your client will be displayed this way: Client “x”: /”ip”.

7.	After pressing the Ready button, two new windows will open. 

8.	The window SPEED SCHEMA will show the changing speed of the different fans. It works with red color circles. When a specific fan accelerates, the circle in its position will become bigger. The same happens with all the other positions of the fans. This window is just a visual representation of the speed of all the active fans.

9.	Moreover, the FAN CONTROL window is the one used to control the fans. There are two different ways to control them. 
    - Manually: At the left side of the window, you can see the Fans schema, where you will be able to select your active fans (comment2). Once you select one of them, on the upper right part, you will see its current speed. With the scrolling bar you can set the speed you want for the fan. Once set, click on the Update button to send the speed to the fan. The fan will automatically start spinning at that speed and on the SPEED SCHEMA you will see a change in its circle dimension (depending on the new speed). To stop all the fans, click on the Stop All button.
    - By a functionality file: If you don’t want to manually set every fan you can also use a functionality file. The file must me a .xlsx. The first column is the for the time and the next columns are for the fans speed. You have some examples on the software/functionalities folder. To use it, click on the Add Functionality Button on the window. Drag and drop the file. Once added you will see its name below. Now you have two options:
        - Real execution: If you click on execute preview the fans will receive their new speeds and the functionality will execute in real time. To stop it you can click on Stop Execution. During this execution, the manual fan control will be disabled.
        - Preview execution: If you click on Execute Preview the fans will not change. A window like SPEED SCHEMA will open, and there you will be able to see what will happen to the fans if you execute the functionality. To close it just close the window.

10.	Apart from this, you can also click on Show Pressure to see the pressure values on real time on a window. For this, you will need and specific pressure sensor. The first time you open it you will have to introduce its IP address and its port number.

11.	Before closing the program don’t forget to Stop All the fans. Otherwise, they will keep the last speed value you gave them, and as soon as they connect to current, they will spin at this speed.
 
### Comments
- Comment 1
The default server port is the 125. In case you want to change it, go to the serverCode project, in the folder “x”. Open the class “Main.java”. In the line 14, replace “125” with the port you wish.

- Comment 2
Once pressed the ready button. The software will check how many clients are connected to the server to display their fans. If only one client is connected, the only available fans will be from 1 to 6. If there are two, from 1 to 12, and so on. The available fans appear in white color and can be clicked, while the not available ones appear in red color and cannot be clicked.

