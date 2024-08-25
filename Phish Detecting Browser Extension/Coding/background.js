chrome.webRequest.onBeforeRequest.addListener(
    function(details) {
      // Extract URL
      const url = new URL(details.url).hostname;
      
      // Call the phishing detection API
      fetch(`http://localhost:5000/check?url=${encodeURIComponent(url)}`)
        .then(response => response.json())
        .then(data => {
          if (data.isPhishing) {
            chrome.browserAction.setIcon({path: "icons/phishing.png"});
            chrome.browserAction.setPopup({popup: "phishing_popup.html"});
          } else {
            chrome.browserAction.setIcon({path: "icons/secure.png"});
            chrome.browserAction.setPopup({popup: "secure_popup.html"});
          }
        })
        .catch(error => console.error('Error:', error));
    },
    {urls: ["<all_urls>"]},
    ["blocking"]
  );
  