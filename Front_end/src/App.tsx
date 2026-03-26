import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

type Role = 'user' | 'assistant'

interface Message {
  id: string
  role: Role
  content: string
  createdAt: string
}

interface ChatResponse {
  thread_id: string
  message: string
  mode: string
}

interface HealthResponse {
  status: string
  mode: string
  model: string
}

const suggestions = [
  '帮我写一个本周工作总结模板',
  '解释一下 FastAPI 和 Flask 的区别',
  '给我做一个产品需求文档的大纲',
]

const createThreadId = () => `thread_${Date.now()}`

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [threadId, setThreadId] = useState(createThreadId)
  const [modeLabel, setModeLabel] = useState('正在连接后端')
  const [modelLabel, setModelLabel] = useState('未获取')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // 启动时读一次后端状态，让页面知道当前是演示模式还是模型模式。
    fetch('/api/health')
      .then(response => response.json() as Promise<HealthResponse>)
      .then(data => {
        setModeLabel(data.mode === 'openai' ? '模型模式' : '演示模式')
        setModelLabel(data.model)
      })
      .catch(() => {
        setModeLabel('后端未连接')
        setModelLabel('不可用')
      })
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isSending])

  useEffect(() => {
    const textarea = textareaRef.current

    if (!textarea) {
      return
    }

    // 输入框高度跟着内容走，体验会更像聊天产品。
    textarea.style.height = '0px'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 220)}px`
  }, [input])

  const appendMessage = (role: Role, content: string) => {
    const nextMessage: Message = {
      id: `${role}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      role,
      content,
      createdAt: new Date().toISOString(),
    }

    setMessages(current => [...current, nextMessage])
  }

  const sendMessage = (content: string) => {
    const trimmed = content.trim()

    if (!trimmed || isSending) {
      return
    }

    appendMessage('user', trimmed)
    setInput('')
    setIsSending(true)

    fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: trimmed,
        thread_id: threadId,
      }),
    })
      .then(async response => {
        const data = await response.json() as ChatResponse

        if (!response.ok) {
          return Promise.reject(new Error('消息发送失败'))
        }

        setThreadId(data.thread_id)
        setModeLabel(data.mode === 'openai' ? '模型模式' : '演示模式')
        appendMessage('assistant', data.message)
      })
      .catch(() => {
        appendMessage('assistant', '当前没有拿到可用响应，请检查后端是否启动。')
      })
      .finally(() => {
        setIsSending(false)
      })
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    sendMessage(input)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendMessage(input)
    }
  }

  const handleNewChat = () => {
    // 新会话会清掉当前线程，避免旧上下文继续带入。
    fetch(`/api/conversations/${threadId}`, {
      method: 'DELETE',
    }).finally(() => {
      setMessages([])
      setInput('')
      setThreadId(createThreadId())
    })
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand-card">
          <span className="brand-badge">AI</span>
          <div>
            <h1>Agent Console</h1>
            <p>先做成纯聊天平台，后续再接模型与工具。</p>
          </div>
        </div>

        <button className="new-chat-button" onClick={handleNewChat} type="button">
          新建会话
        </button>

        <section className="sidebar-panel">
          <span className="panel-label">运行状态</span>
          <p>{modeLabel}</p>
          <p>{modelLabel}</p>
        </section>

        <section className="sidebar-panel">
          <span className="panel-label">建议开场</span>
          <div className="prompt-list">
            {suggestions.map(item => (
              <button
                className="prompt-chip"
                key={item}
                onClick={() => setInput(item)}
                type="button"
              >
                {item}
              </button>
            ))}
          </div>
        </section>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">Chat Workspace</p>
            <h2>像 ChatGPT 一样直接对话</h2>
          </div>
          <div className="thread-card">
            <span>Thread</span>
            <code>{threadId}</code>
          </div>
        </header>

        <section className="conversation">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p className="empty-kicker">准备好了</p>
              <h3>先发一条消息，我们把平台主链路跑顺。</h3>
              <p>
                现在默认是纯聊天版，没有接 MCP。
                等前后端稳定后，再把 RAG 作为独立能力接回来会更稳。
              </p>
            </div>
          ) : (
            <div className="message-list">
              {messages.map(message => (
                <article className={`message-card ${message.role}`} key={message.id}>
                  <div className="avatar">{message.role === 'user' ? '你' : 'AI'}</div>
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>{message.role === 'user' ? '你' : '助手'}</strong>
                      <span>{formatTime(message.createdAt)}</span>
                    </div>
                    {message.role === 'assistant' ? (
                      <div className="markdown-body">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <div className="plain-body">{message.content}</div>
                    )}
                  </div>
                </article>
              ))}
              {isSending && (
                <article className="message-card assistant pending">
                  <div className="avatar">AI</div>
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>助手</strong>
                      <span>生成中</span>
                    </div>
                    <div className="typing-dots">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </article>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        <footer className="composer-wrap">
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              className="composer-input"
              disabled={isSending}
              onChange={event => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的消息，按 Enter 发送"
              rows={1}
              value={input}
            />
            <button className="send-button" disabled={!input.trim() || isSending} type="submit">
              发送
            </button>
          </form>
        </footer>
      </main>
    </div>
  )
}

export default App
