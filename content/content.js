
// Listen for messages from popup.js or background.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "GET_VISIBLE_PLACES") {
    const result = extractVisiblePlaces(); // your extraction logic here
    sendResponse({ places: result });
  }

  return true; // Keep sendResponse alive for async if needed
});

// Example function to extract place titles from the page
function extractVisiblePlaces() {
  const placeElements = document.querySelectorAll('[aria-label][role="article"]');
  const places = [];

  placeElements.forEach(el => {
    const title = el.getAttribute('aria-label');
    if (title) {
      places.push(title);
    }
  });

  return places;
}
