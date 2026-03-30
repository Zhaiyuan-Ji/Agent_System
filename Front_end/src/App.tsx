import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

type Role = 'user' | 'assistant'

interface Message {
  id: string
  role: Role
  content: string
  createdAt: string
  toolCalls?: ToolCall[]
}

interface ToolCall {
  id: string
  tool: string
  args: string
  status: 'calling' | 'completed'
  resultPreview?: string
}

interface ChatResponse {
  thread_id: string
  message: string
}

interface HealthResponse {
  status: string
}

interface PromptMessage {
  id: string | null
  type: string
  role: string
  content: string
}

interface ContextSection {
  key: string
  title: string
  description: string
  messages: PromptMessage[]
}

interface ContextResponse {
  thread_id: string
  current_question: PromptMessage | null
  current_answer: PromptMessage[]
  sections: ContextSection[]
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

function getContextLabel(role: string) {
  if (role === 'system') {
    return '系统消息'
  }

  if (role === 'assistant') {
    return '助手消息'
  }

  if (role === 'tool') {
    return '工具结果'
  }

  return '用户消息'
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [threadId, setThreadId] = useState(createThreadId)
  const [statusLabel, setStatusLabel] = useState('正在连接后端')
  const [showContextPanel, setShowContextPanel] = useState(false)
  const [isLoadingContext, setIsLoadingContext] = useState(false)
  const [contextData, setContextData] = useState<ContextResponse | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const refreshHealth = () => {
    fetch('/api/health')
      .then(response => response.json() as Promise<HealthResponse>)
      .then(() => {
        setStatusLabel('已连接')
      })
      .catch(() => {
        setStatusLabel('后端未连接')
      })
  }

  const refreshContext = (draft = '') => {
    setIsLoadingContext(true)

    fetch(`/api/context/${threadId}?draft=${encodeURIComponent(draft)}`)
      .then(response => response.json() as Promise<ContextResponse>)
      .then(data => {
        setContextData(data)
      })
      .finally(() => {
        setIsLoadingContext(false)
      })
  }

  useEffect(() => {
    refreshHealth()

    const timer = window.setInterval(() => {
      refreshHealth()
    }, 5000)

    return () => {
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isSending])

  useEffect(() => {
    const textarea = textareaRef.current

    if (!textarea) {
      return
    }

    textarea.style.height = '0px'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 220)}px`
  }, [input])

  useEffect(() => {
    if (showContextPanel) {
      refreshContext()
    }
  }, [showContextPanel, threadId])

  const appendMessage = (role: Role, content: string) => {
    const nextMessage: Message = {
      id: `${role}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      role,
      content,
      createdAt: new Date().toISOString(),
    }

    setMessages(current => [...current, nextMessage])
  }

  const sendMessage = async (content: string) => {
    const trimmed = content.trim()

    if (!trimmed || isSending) {
      return
    }

    appendMessage('user', trimmed)
    setInput('')
    setIsSending(true)

    const assistantMsgId = `${'assistant'}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    setMessages(current => [...current, {
      id: assistantMsgId,
      role: 'assistant' as Role,
      content: '',
      createdAt: new Date().toISOString(),
      toolCalls: [],
    }])

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: trimmed,
          thread_id: threadId,
        }),
      })

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let fullContent = ''
      let currentThreadId = threadId

      while (reader) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value)
        const lines = text.split('\n').filter(l => l.trim())

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'text') {
                fullContent += data.content
                setMessages(current => current.map(msg =>
                  msg.id === assistantMsgId
                    ? { ...msg, content: fullContent }
                    : msg
                ))
              } else if (data.type === 'tool_call') {
                const toolCall: ToolCall = {
                  id: `tc_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
                  tool: data.tool,
                  args: data.args,
                  status: 'calling',
                }
                setMessages(current => current.map(msg =>
                  msg.id === assistantMsgId
                    ? { ...msg, toolCalls: [...(msg.toolCalls || []), toolCall] }
                    : msg
                ))
              } else if (data.type === 'tool_result') {
                setMessages(current => current.map(msg => {
                  if (msg.id !== assistantMsgId) return msg
                  const toolCalls = msg.toolCalls || []
                  if (toolCalls.length === 0) {
                    const newToolCall: ToolCall = {
                      id: `tc_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
                      tool: data.tool || 'unknown',
                      args: '{}',
                      status: 'completed',
                      resultPreview: data.preview,
                    }
                    return { ...msg, toolCalls: [newToolCall] }
                  }
                  const lastIndex = toolCalls.length - 1
                  const updatedToolCalls = [...toolCalls]
                  updatedToolCalls[lastIndex] = {
                    ...updatedToolCalls[lastIndex],
                    status: 'completed',
                    resultPreview: data.preview,
                  }
                  return { ...msg, toolCalls: updatedToolCalls }
                }))
              } else if (data.type === 'done') {
                currentThreadId = data.thread_id || currentThreadId
              }
            } catch (e) {
              // ignore parse errors
            }
          }
        }
      }

      setThreadId(currentThreadId)
      setStatusLabel('已连接')

      if (showContextPanel) {
        window.setTimeout(() => {
          refreshContext()
        }, 200)
      }
    } catch (e) {
      console.error('Stream error:', e)
      refreshHealth()
      setMessages(current => current.map(msg =>
        msg.id === assistantMsgId
          ? { ...msg, content: '当前没有拿到可用响应，请检查后端是否正常启动。' }
          : msg
      ))
    } finally {
      setIsSending(false)
    }
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
    fetch(`/api/conversations/${threadId}`, {
      method: 'DELETE',
    }).finally(() => {
      setMessages([])
      setInput('')
      setThreadId(createThreadId())
      setContextData(null)
    })
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand-card">
          <span className="brand-badge">AI</span>
          <div>
            <h1>Agent Console</h1>
            <p>左侧是固定控制区，右侧分别滚动聊天区和上下文区。</p>
          </div>
        </div>

        <button className="new-chat-button" onClick={handleNewChat} type="button">
          新建会话
        </button>

        <button
          className="secondary-button"
          onClick={() => {
            setShowContextPanel(current => !current)
          }}
          type="button"
        >
          {showContextPanel ? '隐藏实时上下文' : '查看实时上下文'}
        </button>

        <section className="sidebar-panel">
          <span className="panel-label">运行状态</span>
          <p>{statusLabel}</p>
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
            <h2>Are you ready, JZY?------ Let's Go!</h2>
          </div>
          <div className="header-actions">
            <div className="thread-card">
              <span>Thread</span>
              <code>{threadId}</code>
            </div>
            {showContextPanel && (
              <button
                className="context-refresh-button"
                onClick={() => refreshContext(input.trim())}
                type="button"
              >
                {isLoadingContext ? '刷新中...' : '刷新上下文'}
              </button>
            )}
          </div>
        </header>

        {showContextPanel && (
          <section className="context-panel">
            <div className="context-panel-header">
              <div>
                <p className="eyebrow">Live Prompt</p>
                <h3>当前实际送给模型的上下文</h3>
              </div>
            </div>
            <p className="context-tip">这里明确区分当前问题、当前轮回复和过去上下文，不再把它们混在一起。</p>

            <div className="context-list">
              <article className="context-current-question">
                <div className="context-item-meta">
                  <strong>当前问题</strong>
                </div>
                <pre>{contextData?.current_question?.content || '当前还没有可以作为当前问题展示的内容。'}</pre>
              </article>

              {Boolean(contextData?.current_answer.length) && (
                <section className="context-section">
                  <div className="context-section-header">
                    <strong>当前轮回复</strong>
                    <p>这部分是模型对当前问题的最近回复。</p>
                  </div>
                  <div className="context-section-list">
                    {contextData?.current_answer.map((message, index) => (
                      <article className={`context-item ${message.role}`} key={`${message.id ?? 'current'}_${index}`}>
                        <div className="context-item-meta">
                          <strong>{getContextLabel(message.role)}</strong>
                        </div>
                        <pre>{message.content}</pre>
                      </article>
                    ))}
                  </div>
                </section>
              )}

              {(contextData?.sections ?? []).map(section => (
                <section className="context-section" key={section.key}>
                  <div className="context-section-header">
                    <strong>{section.title}</strong>
                    <p>{section.description}</p>
                  </div>
                  <div className="context-section-list">
                    {section.messages.map((message, index) => (
                      <article className={`context-item ${message.role}`} key={`${message.id ?? section.key}_${index}`}>
                        <div className="context-item-meta">
                          <strong>{getContextLabel(message.role)}</strong>
                        </div>
                        <pre>{message.content}</pre>
                      </article>
                    ))}
                  </div>
                </section>
              ))}

              {!isLoadingContext && (!contextData || (contextData.sections.length === 0 && !contextData.current_question)) && (
                <div className="context-empty">当前线程还没有可展示的上下文。</div>
              )}
            </div>
          </section>
        )}

        <section className="conversation">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p className="empty-kicker">准备好了</p>
              <h3>先发一条消息，我们把对话链路跑顺。</h3>
              <p>打开实时上下文面板后，你可以清楚看到当前问题和过去上下文是如何分开的。</p>
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
                    {message.role === 'assistant' && message.toolCalls && message.toolCalls.length > 0 && (
                      <div className="tool-calls">
                        <div className="tool-calls-header">
                          <span className="tool-calls-icon">🔧</span>
                          <strong>调用了 {message.toolCalls.length} 个工具</strong>
                        </div>
                        <div className="tool-calls-list">
                          {message.toolCalls.map((tc) => (
                            <div key={tc.id} className={`tool-call-item ${tc.status}`}>
                              <div className="tool-call-header">
                                <span className="tool-name">
                                  {tc.status === 'calling' ? '⏳' : '✅'} {tc.tool}
                                </span>
                              </div>
                              <div className="tool-call-args">
                                <span className="tool-args-label">参数:</span>
                                <code>{tc.args}</code>
                              </div>
                              {tc.resultPreview && (
                                <div className="tool-call-result">
                                  <span className="tool-result-label">结果预览:</span>
                                  <pre>{tc.resultPreview}</pre>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
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
