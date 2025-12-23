import React, { useState } from 'react';
import './App.css';

function App() {
  // State variables (data that can change)
  const [query, setQuery] = useState('');  // stores what user types
  const [results, setResults] = useState(null);  // stores search results
  const [loading, setLoading] = useState(false);  // shows loading state
  const [error, setError] = useState(null);  // stores any errors

  // Function that runs when user submits the search
  const handleSearch = async (e) => {
    e.preventDefault();  // prevents page reload

    if (!query.trim()) {
      return;  // don't search if query is empty
    }

    setLoading(true);  // show loading indicator
    setError(null);  // clear any previous errors

    try {
      // Make API call to backend
      const response = await fetch('http://localhost:8000/rag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          k: 5,
          corpus: 'msmarco',
          method: 'rerank'
        })
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setResults(data);  // store the results
    } catch (err) {
      setError('Failed to search. Make sure the backend is running on port 8000.');
      console.error(err);
    } finally {
      setLoading(false);  // hide loading indicator
    }
  };

  return (
    <div className="App">
      <div className="container">
        <h1>RAG Search</h1>

        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your question..."
            className="search-input"
          />
          <button type="submit" disabled={loading} className="search-button">
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {results && (
          <div className="results">
            <div className="answer-section">
              <h2>Answer</h2>
              <p className="answer">{results.answer}</p>
            </div>

            <div className="chunks-section">
              <h2>Retrieved Chunks ({results.chunks.length})</h2>
              {results.chunks.map((chunk, index) => (
                <div key={index} className="chunk">
                  <div className="chunk-header">
                    <span className="chunk-rank">#{index + 1}</span>
                    <span className="chunk-score">Score: {chunk.score.toFixed(4)}</span>
                    <span className="chunk-id">Doc {chunk.doc_id} / Chunk {chunk.chunk_id}</span>
                  </div>
                  <p className="chunk-text">{chunk.chunk_text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
