// Listen for a click event on the "Search" button
document.getElementById("sendBtn").addEventListener("click", async () => {
    // Get the user's input from the text field and trim whitespace
    const query = document.getElementById("queryInput").value.trim();
    // Get the output div element where we will show results or messages
    const output = document.getElementById("output");
  
    // If the input is empty, show a message and stop
    if (!query) {
      output.textContent = "Please enter a query.";
      return; // exit the function early
    }
  
    // Show a temporary status message while we fetch data
    output.textContent = "Searching...";
  
    try {
      // Send a GET request to your backend server with the user's query as a URL parameter
      const response = await fetch(`http://127.0.0.1:8000/api/search?query=${encodeURIComponent(query)}`).then(res => {res.json()});
      // Parse the JSON response from the server
      const data = await response.json();
  
      // Check if the response has results and is not empty
      if (data.results && data.results.length > 0) {
        // Map over the results and extract the 'name' property, then join them with newlines
        output.textContent = data.results.map(place => `• ${place.name}`).join("\n");
      } else {
        // If no results found, show this message
        output.textContent = "No results found.";
      }
    } catch (error) {
      // If there's an error (e.g., network error), log it and show an error message
      console.error(error);
      output.textContent = "Error fetching data.";
    }
  });  