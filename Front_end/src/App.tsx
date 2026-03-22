import { useState, useRef, useEffect } from 'react'
import './App.css'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [threadId] = useState(() => `thread_${Date.now()}`)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          thread_id: threadId
        })
      })

      if (!response.ok) throw new Error('请求失败')

      const data = await response.json()

      const assistantMessage: Message = {
        id: `assistant_${Date.now()}`,
        role: 'assistant',
        content: data.message,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        role: 'assistant',
        content: '抱歉，发生了错误，请稍后重试。',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }

  const handleClearConversation = () => {
    setMessages([])
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">AI</div>
            <span className="logo-text">Agent 助手</span>
          </div>
          <div className="header-actions">
            <button className="clear-btn" onClick={handleClearConversation}>
              清除对话
            </button>
          </div>
        </div>
      </header>

      <main className="chat-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-icon">🤖</div>
            <h1>欢迎使用 AI Agent 助手</h1>
            <p>我是基于 Milvus 混合检索的智能助手，可以帮助你回答各种问题。</p>
            <div className="suggestions">
              <div className="suggestion-card" onClick={() => setInput('信号处理方法主要有哪几种？')}>
                <span className="suggestion-icon">📡</span>
                <span>信号处理方法</span>
              </div>
              <div className="suggestion-card" onClick={() => setInput('5G毫米波通信有什么特点？')}>
                <span className="suggestion-icon">📱</span>
                <span>5G毫米波通信</span>
              </div>
              <div className="suggestion-card" onClick={() => setInput('电磁波在大气中的传播特性')}>
                <span className="suggestion-icon">📡</span>
                <span>电磁波传播特性</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map(message => (
              <div key={message.id} className={`message ${message.role}`}>
                <div className="message-avatar">
                  {message.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-header">
                    <span className="message-role">
                      {message.role === 'user' ? '你' : 'AI 助手'}
                    </span>
                    <span className="message-time">{formatTime(message.timestamp)}</span>
                  </div>
                  <div className="message-text">
                    {message.content.split('\n').map((line, i) => (
                      <p key={i}>{line}</p>
                    ))}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <footer className="input-container">
        <form className="input-form" onSubmit={handleSubmit}>
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              className="input-field"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入消息..."
              rows={1}
              disabled={isLoading}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || isLoading}
            >
              {isLoading ? (
                <span className="loading-spinner"></span>
              ) : (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
            </button>
          </div>
          <p className="input-hint">按 Enter 发送，Shift + Enter 换行</p>
        </form>
      </footer>
    </div>
  )
}

export default App