from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Dummy function to check URL
def check_phishing(url):
    # Implement your phishing detection logic here
    # For example, use machine learning models or third-party APIs
    # This is a placeholder for demonstration purposes
    return url in ["malicious-site.com", "phishing-example.com"]

@app.route('/check')
def check():
    url = request.args.get('url')
    is_phishing = check_phishing(url)
    return jsonify({'isPhishing': is_phishing})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
