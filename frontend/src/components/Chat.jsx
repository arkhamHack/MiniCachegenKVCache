import React, { useState, useRef, useEffect } from 'react';
import MessageContent from './MessageContent';
import '../styles/Chat.css';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [cachegenEnabled, setCachegenEnabled] = useState(true);
  const [metrics, setMetrics] = useState({
    withCache: { avgLatency: null, memoryUsage: null },
    withoutCache: { avgLatency: null, memoryUsage: null }
  });
  const chatContainerRef = useRef(null);
  const wsRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const formatTimestamp = () => {
    const now = new Date();
    return `${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')}`;
  };

  const addMessage = (role, content, metadata = {}) => {
    setMessages(prev => [...prev, {
      id: Date.now(),
      role,
      content,
      timestamp: formatTimestamp(),
      metadata
    }]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = input;
    setInput('');
    addMessage('user', userMessage);
    setIsLoading(true);

    try {
      // Get performance metrics
      const metricsResponse = await fetch('http://localhost:8000/metrics');
      const currentMetrics = await metricsResponse.json();
      setMetrics(currentMetrics);

      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: userMessage,
          stream: true,
          use_cachegen: cachegenEnabled
        }),
      });

      let assistantMessage = '';
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // Add initial assistant message
      const assistantId = Date.now();
      addMessage('assistant', '', { id: assistantId });

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const token = line.slice(6);
            assistantMessage += token;
            // Update the last assistant message
            setMessages(prev => prev.map(msg => 
              msg.id === assistantId 
                ? { ...msg, content: assistantMessage }
                : msg
            ));
          }
        }
      }
    } catch (error) {
      console.error('Generation error:', error);
      addMessage('system', 'Error: Failed to generate response');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="chat-title">
          <h1>CacheGen</h1>
          <div className="cachegen-stats">
            <div className="cachegen-toggle">
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={cachegenEnabled}
                  onChange={(e) => setCachegenEnabled(e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
              <span>Enable CacheGen optimization</span>
            </div>
            
            <div className="performance-metrics">
              <div className="metric-group">
                <h4>With CacheGen:</h4>
                <div className="metric">
                  <span>Avg. Latency:</span>
                  <span>{metrics.withCache.avgLatency ? `${metrics.withCache.avgLatency.toFixed(2)}ms` : 'N/A'}</span>
                </div>
                <div className="metric">
                  <span>Memory Usage:</span>
                  <span>{metrics.withCache.memoryUsage ? `${metrics.withCache.memoryUsage.toFixed(2)}MB` : 'N/A'}</span>
                </div>
              </div>
              
              <div className="metric-group">
                <h4>Without CacheGen:</h4>
                <div className="metric">
                  <span>Avg. Latency:</span>
                  <span>{metrics.withoutCache.avgLatency ? `${metrics.withoutCache.avgLatency.toFixed(2)}ms` : 'N/A'}</span>
                </div>
                <div className="metric">
                  <span>Memory Usage:</span>
                  <span>{metrics.withoutCache.memoryUsage ? `${metrics.withoutCache.memoryUsage.toFixed(2)}MB` : 'N/A'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="chat-messages" ref={chatContainerRef}>
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.role}`}>
            <div className="message-content">
              {message.role === 'assistant' && (
                <div className="assistant-icon">🤖</div>
              )}
              {message.role === 'user' && (
                <div className="user-icon">👤</div>
              )}
              <div className="message-bubble">
                <MessageContent 
                  content={message.content || (isLoading && message.role === 'assistant' ? '...' : '')}
                />
              </div>
            </div>
            <div className="message-timestamp">{message.timestamp}</div>
            {message.role === 'assistant' && (
              <div className="message-metadata">
                {cachegenEnabled && (
                  <span className="cachegen-badge">
                    CacheGen Enabled
                  </span>
                )}
                <span className="response-time">
                  Response delay: {message.metadata?.responseTime || '0.00'} seconds
                </span>
              </div>
            )}
          </div>
        ))}
      </div>

      <form className="chat-input" onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? '...' : '➤'}
        </button>
      </form>
    </div>
  );
};

export default Chat;
