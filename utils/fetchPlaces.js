// Dynamically load the Google Maps JS API with Places library
export async function loadGoogleMapsApi(apiKey) {
    return new Promise((resolve, reject) => {
      if (window.google && window.google.maps) {
        resolve(window.google);
      } else {
        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places`;
        script.async = true;
        script.defer = true;
        script.onload = () => resolve(window.google);
        script.onerror = () => reject('Failed to load Google Maps API');
        document.head.appendChild(script);
      }
    });
  }
  
  // Fetch nearby places using Google Places API
  export async function fetchNearbyPlaces(apiKey, location, radius = 1500, type = 'tourist_attraction') {
    const google = await loadGoogleMapsApi(apiKey);
  
    return new Promise((resolve, reject) => {
      const dummyMap = new google.maps.Map(document.createElement('div')); // required by PlacesService
      const service = new google.maps.places.PlacesService(dummyMap);
  
      const request = {
        location: new google.maps.LatLng(location.lat, location.lng),
        radius: radius,
        type: type
      };
  
      service.nearbySearch(request, (results, status) => {
        if (status === google.maps.places.PlacesServiceStatus.OK) {
          const processed = results.map(place => ({
            name: place.name,
            rating: place.rating,
            address: place.vicinity,
            location: place.geometry?.location?.toJSON?.(),
            photoUrl: place.photos?.[0]?.getUrl?.({ maxWidth: 400 }) || null,
            placeId: place.place_id
          }));
          resolve(processed);
        } else {
          reject(`Places API error: ${status}`);
        }
      });
    });
  }
  