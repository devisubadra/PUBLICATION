document.addEventListener('DOMContentLoaded', function() {
    const messageElement = document.getElementById('message');
    
    chrome.storage.local.get(['isPhishing'], function(result) {
      if (result.isPhishing) {
        messageElement.textContent = "Warning! This site is identified as phishing.";
      } else {
        messageElement.textContent = "This site is safe.";
      }
    });
  });
  