import React, { useState } from 'react';
import { View, TextInput, Button, Alert } from 'react-native';
import WebSocket from 'ws';

const ws = new WebSocket('ws://192.168.1.100:8080'); // Replace with your server IP and port

const App = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = () => {
    fetch('http://192.168.1.100:8080/login', { // Replace with your server IP and port
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    })
    .then(response => response.json())
    .then(data => {
      if (data.token) {
        Alert.alert('Login successful!');
        ws.send('unlock'); // Send command to unlock door
      } else {
        Alert.alert('Login failed');
      }
    })
    .catch(error => {
      console.error(error);
    });
  };

  return (
    <View>
      <TextInput placeholder="Username" value={username} onChangeText={setUsername} />
      <TextInput placeholder="Password" value={password} onChangeText={setPassword} secureTextEntry />
      <Button title="Login" onPress={handleLogin} />
    </View>
  );
};

export default App;
