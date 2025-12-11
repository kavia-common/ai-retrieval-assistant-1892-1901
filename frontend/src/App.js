import React, { useState, useEffect } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import './App.css';
import RAGDemo from './RAGDemo';

// Simple home landing
function Home() {
  return (
    <div className="page">
      <div className="card">
        <h1 className="title">AI Retrieval Assistant</h1>
        <p className="subtitle">Lightweight RAG demo with Ocean Professional styling.</p>
        <p className="description">
          Use the RAG Demo to ingest text, search top matches, and generate an answer using a simple client-side setup.
        </p>
        <div className="actions">
          <Link to="/rag-demo" className="btn primary">Open RAG Demo</Link>
        </div>
      </div>
    </div>
  );
}

// PUBLIC_INTERFACE
function App() {
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // PUBLIC_INTERFACE
  const toggleTheme = () => setTheme(prev => (prev === 'light' ? 'dark' : 'light'));

  return (
    <div className="App">
      <header className="navbar">
        <div className="navbar-left">
          <Link to="/" className="brand">Ocean AI</Link>
          <nav className="nav-links">
            <Link to="/" className="nav-link">Home</Link>
            <Link to="/rag-demo" className="nav-link">RAG Demo</Link>
          </nav>
        </div>
        <div className="navbar-right">
          <button className="theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}>
            {theme === 'light' ? '🌙 Dark' : '☀️ Light'}
          </button>
        </div>
      </header>

      <main className="main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/rag-demo" element={<RAGDemo />} />
        </Routes>
      </main>

      <footer className="footer">
        <span>© 2025 Ocean AI • Lightweight RAG Demo</span>
        <a href="https://reactjs.org" className="footer-link" rel="noreferrer" target="_blank">React Docs</a>
      </footer>
    </div>
  );
}

export default App;
