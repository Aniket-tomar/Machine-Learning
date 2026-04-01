import { useState } from 'react';
import './index.css';

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!review.trim()) {
      setError("Please enter a review first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8502/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review }),
      });

      if (!response.ok) {
        throw new Error('Server responded with an error');
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      setResult(data);
    } catch (err) {
      setError("Failed to connect to the backend server. Please make sure api.py is running on port 8502.");
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (type) => {
    if (type === 'positive') {
      setReview("This movie was absolutely fantastic with brilliant acting, a gripping storyline, and breathtaking visuals. A true masterpiece!");
    } else {
      setReview("This was the worst movie I have ever watched. The plot was non-existent, the acting was wooden, and I felt like I wasted my time.");
    }
    setError(null);
    setResult(null);
  };

  return (
    <div className="app-container">
      <div className="glass-card">
        <header className="header">
          <h1>IMDB Sentiment Vectorizer</h1>
          <p>Cutting-edge classification wrapped in elegant design.</p>
        </header>

        <div className="examples-container">
          <button className="btn-example" onClick={() => handleExample('positive')}>
            ✨ Positive Example
          </button>
          <button className="btn-example" onClick={() => handleExample('negative')}>
            🌧️ Negative Example
          </button>
        </div>

        <div className="input-container">
          <textarea 
            placeholder="Type your movie review here..."
            value={review}
            onChange={(e) => {
              setReview(e.target.value);
              if (error) setError(null);
            }}
          />
        </div>

        <button 
          className="btn-predict" 
          onClick={handlePredict}
          disabled={loading || !review.trim()}
        >
          {loading ? (
            <><div className="loader"></div> Processing...</>
          ) : (
            'Analyze Sentiment'
          )}
        </button>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {result && (
          <div className={`result-card ${result.prediction === 1 ? 'positive' : 'negative'}`}>
            <div className="result-icon">
              {result.prediction === 1 ? '😊' : '😞'}
            </div>
            <div className="result-title">
              {result.prediction === 1 ? 'Positive Review' : 'Negative Review'}
            </div>
            
            {result.confidence !== null && (
              <div className="confidence-container">
                <div className="confidence-label">
                  <span>Confidence Score</span>
                  <span>{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
