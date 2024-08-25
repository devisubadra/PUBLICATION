const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const users = { // Example user database
  'user1': 'password1'
};

app.use(express.json());

app.post('/login', (req, res) => {
  const { username, password } = req.body;
  if (users[username] && users[username] === password) {
    // Send a token or authentication response
    res.json({ token: 'example_token' });
  } else {
    res.status(401).send('Unauthorized');
  }
});

wss.on('connection', ws => {
  ws.on('message', message => {
    console.log('Received message: %s', message);
    // Broadcast message to all clients (or specific ones)
    wss.clients.forEach(client => {
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  });
});

server.listen(8080, () => {
  console.log('Server is running on port 8080');
});
