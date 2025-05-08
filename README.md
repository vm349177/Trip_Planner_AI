# Trip_Planner_AI
```
trip-planner-ai/
├── backend/
│   ├── main.py                # FastAPI server
│   ├── routes/
│   │   └── planner.py         # Itinerary generation logic
│   ├── services/
│   │   ├── openai_agent.py    # Chatbot interaction with GPT-4
│   │   ├── places_api.py      # Google Places or OSM integration
│   │   └── router.py          # Route optimization logic
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── MapView.jsx    # Leaflet or Mapbox display
│   │   │   ├── Sidebar.jsx    # Time-sorted place list
│   │   ├── pages/
│   │   │   └── Home.jsx
│   │   └── App.jsx
│   └── package.json
├── .env                       # Store API keys
└── README.md
```
