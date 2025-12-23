# RAG Search UI

Simple React frontend for the RAG pipeline.

## Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

## Running the Application

1. Start the backend server (in a separate terminal):
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

2. Start the frontend (in another terminal):
```bash
cd frontend
npm start
```

The UI will open at http://localhost:3000

## Usage

1. Type your question in the search bar
2. Click "Search" or press Enter
3. View the generated answer and retrieved chunks

## Configuration

To change search settings, edit `src/App.js`:
- `k`: number of results (default: 5)
- `corpus`: "msmarco" or "my_corpus"
- `method`: "dense" or "rerank"
