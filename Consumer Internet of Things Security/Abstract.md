
# Consumer Internet of Things Security: Smart Door Lock System

## Overview

This project involves a Smart Door Lock System using an ESP8266 microcontroller, a Node.js server with WebSocket, and a React Native mobile application. The system allows remote control of a door lock via a mobile app, providing a secure and convenient access management solution.

## Components

1. **ESP8266 Lock Client Code**
   - **File Name:** `esp8266_lock_client.ino`
   - **Description:** Code for the ESP8266 microcontroller to control the door lock. It connects to a WebSocket server and listens for commands to lock or unlock the door.

2. **Server Code**
   - **File Name:** `server.js`
   - **Description:** Node.js server using Express and WebSocket. It manages communication between the mobile app and the ESP8266 client.

3. **Mobile Application Code**
   - **File Name:** `App.js`
   - **Description:** React Native application to control the door lock remotely. It connects to the WebSocket server to send lock/unlock commands.

## Installation

### ESP8266 Lock Client Code

1. Install the Arduino IDE if not already installed.
2. Open `esp8266_lock_client.ino` in the Arduino IDE.
3. Configure the Wi-Fi credentials and WebSocket server URL in the code.
4. Upload the code to the ESP8266 microcontroller.

### Server Code

1. Ensure Node.js is installed on your machine.
2. Install the necessary dependencies:
   ```bash
   npm install express ws
   ```
3. Start the server:
   ```bash
   node server.js
   ```
4. The server will listen for WebSocket connections from the ESP8266 and the mobile app.

### Mobile Application Code

1. Set up a new React Native project or integrate the code into your existing project.
2. Install the necessary dependencies for React Native and WebSocket communication.
3. Replace any placeholder WebSocket server URL with your server’s URL in `App.js`.
4. Run the application on a simulator or a physical device.

## Configuration

- **ESP8266:**
  - Modify `WIFI_SSID`, `WIFI_PASSWORD`, and `WEBSOCKET_SERVER_URL` in `esp8266_lock_client.ino` with your network and server details.

- **Server:**
  - No additional configuration required, but you may adjust server settings in `server.js`.

- **Mobile Application:**
  - Update WebSocket URL in `App.js` to match your server’s address.

## Usage

1. **Start the Server:**
   - Run `server.js` to start the WebSocket server.

2. **Connect the ESP8266:**
   - Ensure the ESP8266 is connected to the same network as the server and has the correct WebSocket URL.

3. **Use the Mobile App:**
   - Open the React Native application.
   - Use the provided UI to send lock/unlock commands.

## Contributing

Feel free to submit pull requests or report issues. Contributions are welcome to improve the functionality and features of this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact DEVI SUBADRA VENKATESAN at devisv25@gmail.com.
