import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react'
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

interface HealthResponse {
  status: string
  model?: string
  base_url?: string
}

interface RuntimeState {
  current_goal?: string
  working_memory?: string[]
  confirmed_facts?: string[]
  active_constraints?: string[]
  open_loops?: string[]
  active_evidence_refs?: string[]
  recent_context_summary?: string
  [key: string]: unknown
}

interface ContextSection {
  label?: string
  kind?: string
  value?: unknown
  preview?: string
  estimated_tokens?: number
  selected?: boolean
  message_type?: string
  [key: string]: unknown
}

interface ContextGroup {
  group_id?: string
  label?: string
  kind?: string
  summary?: string
  estimated_tokens?: number
  selected?: boolean
  sections?: ContextSection[]
  [key: string]: unknown
}

interface BackendModelCallRecord {
  thread_id: string
  call_id: string
  call_index: number
  sequence: number
  phase: string
  purpose: string
  input_context: ContextGroup[]
  state_snapshot?: RuntimeState | null
  output?: Record<string, unknown>
  raw_think?: string | null
  started_at?: string
  completed_at?: string | null
}

interface ContextResponse {
  thread_id: string
  model_calls?: BackendModelCallRecord[]
}

const createThreadId = () => `thread_${Date.now()}`

const suggestions = [
  '帮我检索几篇关于毫米波雷达干扰抑制的论文',
  '解释一下 FastAPI 和 Flask 的区别',
  '请使用检索工具帮我检索毫米波雷达干扰抑制相关论文，并给出结果',
]

const DEMO_MODE =
  import.meta.env.VITE_DEMO_MODE === 'true' || new URLSearchParams(window.location.search).get('demo') === '1'

const sleep = (ms: number) => new Promise(resolve => window.setTimeout(resolve, ms))

function formatDisplayContent(content: unknown) {
  if (typeof content === 'string') return content
  try {
    return JSON.stringify(content, null, 2)
  } catch {
    return String(content)
  }
}

function getOutputType(call: BackendModelCallRecord | null) {
  return String(call?.output?.type ?? 'waiting')
}

function getOutputSummary(call: BackendModelCallRecord | null) {
  const output = call?.output ?? {}
  return String(output.summary ?? output.answer ?? '等待模型输出')
}

function getToolName(call: BackendModelCallRecord | null) {
  const toolCalls = call?.output?.tool_calls
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) return ''
  const first = toolCalls[0] as Record<string, unknown>
  return String(first.name ?? '')
}

function displayCallId(call: BackendModelCallRecord) {
  return `call_${call.call_index}`
}

function normalizeContextGroups(call: BackendModelCallRecord | null): ContextGroup[] {
  if (!call) return []
  return (call.input_context ?? []).map((item, index) => {
    if (Array.isArray(item.sections)) return item
    return {
      group_id: `legacy_${index}`,
      label: String(item.label ?? `Context Block ${index + 1}`),
      kind: String(item.kind ?? 'memory'),
      summary: String(item.preview ?? item.value ?? '').slice(0, 80),
      estimated_tokens: Number(item.estimated_tokens ?? 0),
      selected: Boolean(item.selected ?? true),
      sections: [item as ContextSection],
    }
  })
}

function stateSummary(state: RuntimeState | null | undefined) {
  if (!state) return '未捕获状态快照'
  const facts = Array.isArray(state.confirmed_facts) ? state.confirmed_facts.length : 0
  const constraints = Array.isArray(state.active_constraints) ? state.active_constraints.length : 0
  const loops = Array.isArray(state.open_loops) ? state.open_loops.length : 0
  return `${state.current_goal || '未设置目标'} · facts ${facts} · constraints ${constraints} · open ${loops}`
}

function buildDemoModelCalls(threadId: string, request: string): BackendModelCallRecord[] {
  const baseState: RuntimeState = {
    current_goal: request,
    working_memory: ['用户希望看到一次完整的学术检索 Agent 执行过程', '本轮需要展示模型调用、工具调用和上下文装配'],
    confirmed_facts: ['问题属于学术检索任务', '需要外部论文证据支持'],
    active_constraints: ['不编造论文', '优先使用检索工具', '保留可解释的上下文记录'],
    open_loops: ['等待检索工具返回候选论文'],
    active_evidence_refs: [],
    recent_context_summary: '用户发起一次关于毫米波雷达干扰抑制的文献检索请求。',
  }

  return [
    {
      thread_id: threadId,
      call_id: `${threadId}:demo_call_1`,
      call_index: 1,
      sequence: 1,
      phase: 'planning',
      purpose: '理解用户请求并判断是否需要外部工具。',
      input_context: [
        {
          group_id: 'instructions',
          label: 'Input: Instructions',
          kind: 'system',
          summary: '系统规则与工具边界',
          estimated_tokens: 210,
          selected: true,
          sections: [
            {
              label: 'system_prompt',
              kind: 'system',
              value:
                '你是一个严谨的学术检索 Agent。遇到论文、作者、年份、研究方向等问题时，需要优先判断是否需要调用检索工具，不要编造论文。',
            },
          ],
        },
        {
          group_id: 'conversation',
          label: 'Input: Conversation',
          kind: 'dialogue',
          summary: '当前用户请求',
          estimated_tokens: 80,
          selected: true,
          sections: [
            {
              label: 'user_request',
              kind: 'user',
              value: request,
            },
          ],
        },
      ],
      state_snapshot: baseState,
      output: {
        type: 'tool_call',
        summary: '模型判断需要外部证据，发起 hybrid_search。',
        tool_calls: [
          {
            name: 'hybrid_search',
            args: {
              query: 'mmWave radar interference mitigation automotive FMCW radar mutual interference',
              limit: 8,
            },
          },
        ],
      },
      raw_think:
        'Demo raw think：这里展示的是演示模式下的折叠调试内容，用来说明 raw think 只应该在用户主动点击后查看。',
      started_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
    },
    {
      thread_id: threadId,
      call_id: `${threadId}:demo_call_2`,
      call_index: 2,
      sequence: 2,
      phase: 'tool_observation',
      purpose: '读取检索结果，更新证据和任务状态。',
      input_context: [
        {
          group_id: 'tool_evidence',
          label: 'Input: Tool Evidence',
          kind: 'evidence',
          summary: '检索工具返回候选论文',
          estimated_tokens: 460,
          selected: true,
          sections: [
            {
              label: 'tool_message',
              kind: 'tool',
              value:
                '[1] Automotive FMCW Radar Mutual Interference: Survey and Mitigation | 2024\n[2] Deep Learning Based Interference Suppression for mmWave Radar | 2023\n[3] Time-Frequency Methods for Radar Interference Mitigation | 2022',
            },
            {
              label: 'evidence',
              kind: 'retrieval',
              value:
                '检索结果覆盖车载 FMCW 雷达互扰、深度学习干扰抑制、时频域干扰检测与重构等方向。',
            },
          ],
        },
        {
          group_id: 'state',
          label: 'Input: State',
          kind: 'state',
          summary: '工具结果已进入工作记忆',
          estimated_tokens: 140,
          selected: true,
          sections: [
            {
              label: 'working_memory',
              kind: 'state',
              value: [
                '已获得 3 条演示用候选论文证据',
                '需要把论文按研究方向归类，并提示这是在线 demo 的模拟数据',
              ],
            },
          ],
        },
      ],
      state_snapshot: {
        ...baseState,
        confirmed_facts: [...(baseState.confirmed_facts ?? []), '检索工具返回了毫米波雷达干扰抑制相关候选论文'],
        open_loops: ['需要生成面向用户的归纳回答'],
        active_evidence_refs: ['demo-paper-1', 'demo-paper-2', 'demo-paper-3'],
      },
      output: {
        type: 'answer_draft',
        summary: '模型准备把检索结果归纳为最终回答。',
      },
      raw_think:
        'Demo raw think：模型把工具证据转换成可读回答，并保留“这是演示数据”的边界说明。',
      started_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
    },
    {
      thread_id: threadId,
      call_id: `${threadId}:demo_call_3`,
      call_index: 3,
      sequence: 3,
      phase: 'answering',
      purpose: '基于工具证据、状态和上下文生成最终回答。',
      input_context: [
        {
          group_id: 'final_context',
          label: 'Input: Final Context',
          kind: 'assembly',
          summary: '系统规则、用户请求、工具证据、状态快照',
          estimated_tokens: 780,
          selected: true,
          sections: [
            { label: 'system_prompt', kind: 'system', value: '保持严谨，不编造引用，说明证据来源。' },
            { label: 'user_request', kind: 'user', value: request },
            {
              label: 'tool_message',
              kind: 'tool',
              value:
                'hybrid_search 返回了车载 FMCW 互扰、深度学习抑制、时频域检测与重构三类候选研究方向。',
            },
            { label: 'state', kind: 'state', value: '证据已确认，准备输出最终回答。' },
          ],
        },
      ],
      state_snapshot: {
        ...baseState,
        confirmed_facts: [
          ...(baseState.confirmed_facts ?? []),
          '候选论文可按互扰综述、深度学习抑制、时频域重构三类组织',
        ],
        open_loops: [],
        active_evidence_refs: ['demo-paper-1', 'demo-paper-2', 'demo-paper-3'],
      },
      output: {
        type: 'final_answer',
        summary: '输出最终文献调研回答。',
        answer:
          '已经基于演示检索结果整理出 3 个方向：车载 FMCW 雷达互扰综述、深度学习干扰抑制、时频域检测与重构。',
      },
      raw_think:
        'Demo raw think：最终回答阶段只引用已进入工具证据的内容，不额外编造论文。',
      started_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
    },
  ]
}

function buildDemoAnswer() {
  return [
    '这是在线演示模式下的一次 Agent 运行示例。',
    '',
    '我会把这个问题当作学术检索任务处理，并先调用检索工具获取外部证据。演示中返回了 3 类相关方向：',
    '',
    '1. **车载 FMCW 雷达互扰综述**：适合用来了解毫米波雷达互扰的成因、场景和常见缓解路线。',
    '2. **深度学习干扰抑制**：通常关注用神经网络从受干扰回波中恢复目标信息。',
    '3. **时频域检测与重构方法**：更偏信号处理路线，例如先定位受干扰区域，再做插值、重构或滤除。',
    '',
    '你可以点击右侧的 `call_1`、`call_2`、`call_3`，查看每一次模型调用看到的上下文、状态快照和输出。这个在线 demo 使用模拟数据，目的是展示平台交互和 Agent 调试流程；真实本地运行时会连接 Redis、Milvus、embedding 服务和 MiniMax 模型。',
  ].join('\n')
}

function ConversationHud({
  modelCalls,
  toolCallCount,
  latestPhase,
  isSending,
}: {
  modelCalls: BackendModelCallRecord[]
  toolCallCount: number
  latestPhase: string
  isSending: boolean
}) {
  return (
    <section className="conversation-hud">
      <div className="hud-core">
        <i />
        <i />
        <strong>{isSending ? 'LIVE' : 'READY'}</strong>
      </div>
      <div className="hud-card">
        <span>LLM Calls</span>
        <strong>{modelCalls.length}</strong>
      </div>
      <div className="hud-card">
        <span>Tools</span>
        <strong>{toolCallCount}</strong>
      </div>
      <div className="hud-card phase-card">
        <span>Phase</span>
        <strong>{latestPhase}</strong>
      </div>
      <div className="hud-wave">
        <b />
        <b />
        <b />
        <b />
        <b />
      </div>
    </section>
  )
}

function CallFocus({
  call,
  onClose,
  onShowThink,
}: {
  call: BackendModelCallRecord
  onClose: () => void
  onShowThink: () => void
}) {
  const groups = normalizeContextGroups(call)

  return (
    <section className="call-focus">
      <div className="focus-hero">
        <div>
          <span className="kicker">Call Projection</span>
          <h3>{displayCallId(call)}</h3>
          <p>{call.purpose}</p>
        </div>
        <div className="focus-orb">
          <i />
          <strong>{call.phase}</strong>
        </div>
        <button className="ghost-action compact-action" onClick={onClose} type="button">
          返回对话
        </button>
      </div>

      <div className="focus-grid">
        <section className="focus-panel scroll-window">
          <div className="focus-panel-head">
            <span>Input Context</span>
            <strong>{groups.length} groups</strong>
          </div>
          <div className="context-stack">
            {groups.map((group, index) => (
              <details
                className={`context-group ${String(group.kind ?? 'memory')}`}
                key={`${call.call_id}_${group.group_id ?? index}`}
              >
                <summary>
                  <span>{String(group.label ?? `Input Block ${index + 1}`)}</span>
                  <strong>{String(group.summary ?? `${group.sections?.length ?? 0} sections`)}</strong>
                </summary>
                <div className="context-section-list">
                  {(group.sections ?? []).map((section, sectionIndex) => (
                    <details
                      className="context-section"
                      key={`${group.group_id ?? index}_${String(section.label ?? sectionIndex)}`}
                    >
                      <summary>
                        <span>{String(section.label ?? `section_${sectionIndex + 1}`)}</span>
                        <code>{String(section.kind ?? section.message_type ?? 'context')}</code>
                      </summary>
                      <pre>{formatDisplayContent(section.value ?? '')}</pre>
                    </details>
                  ))}
                </div>
              </details>
            ))}
          </div>
        </section>

        <section className="focus-panel scroll-window">
          <div className="focus-panel-head">
            <span>State Snapshot</span>
            <strong>{call.state_snapshot ? 'captured' : 'empty'}</strong>
          </div>
          <details className="inspector-disclosure" open>
            <summary>{stateSummary(call.state_snapshot)}</summary>
            <pre className="data-block">{formatDisplayContent(call.state_snapshot ?? {})}</pre>
          </details>
        </section>

        <section className="focus-panel scroll-window output-focus">
          <div className="focus-panel-head">
            <span>Model Output</span>
            <strong>{getOutputType(call)}</strong>
          </div>
          <details className="inspector-disclosure" open>
            <summary>
              {getToolName(call) ? `${getOutputType(call)} · ${getToolName(call)}` : getOutputSummary(call)}
            </summary>
            <article className="output-card">
              {getToolName(call) && <code>{getToolName(call)}</code>}
              <pre>{formatDisplayContent(call.output ?? {})}</pre>
            </article>
            <button className="think-button" disabled={!call.raw_think} onClick={onShowThink} type="button">
              {call.raw_think ? '查看 raw think' : '未捕获 raw think'}
            </button>
          </details>
        </section>
      </div>
    </section>
  )
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [threadId, setThreadId] = useState(createThreadId)
  const [statusLabel, setStatusLabel] = useState('连接中')
  const [activeModel, setActiveModel] = useState('未知模型')
  const [modelCalls, setModelCalls] = useState<BackendModelCallRecord[]>([])
  const [selectedCallId, setSelectedCallId] = useState<string | null>(null)
  const [showThinkModal, setShowThinkModal] = useState(false)
  const [showContextPanel, setShowContextPanel] = useState(false)
  const [isLoadingContext, setIsLoadingContext] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const conversationEndRef = useRef<HTMLDivElement>(null)
  const scrollSessionRef = useRef<{
    container: HTMLElement
    startX: number
    startY: number
    startScrollTop: number
    startScrollLeft: number
    canScrollX: boolean
    canScrollY: boolean
  } | null>(null)

  const selectedCall = useMemo(
    () => modelCalls.find(call => call.call_id === selectedCallId) ?? null,
    [modelCalls, selectedCallId],
  )
  const latestCall = modelCalls.length ? modelCalls[modelCalls.length - 1] : null
  const toolCallCount = modelCalls.filter(call => getOutputType(call) === 'tool_call').length
  const latestPhase = selectedCall?.phase || latestCall?.phase || (isSending ? 'running' : 'idle')

  const refreshHealth = () => {
    if (DEMO_MODE) {
      setStatusLabel('DEMO')
      setActiveModel('MiniMax-M2.7 demo')
      return
    }

    fetch('/api/health')
      .then(response => response.json() as Promise<HealthResponse>)
      .then(data => {
        setStatusLabel(data.status === 'ok' ? 'ONLINE' : 'DEGRADED')
        if (data.model) setActiveModel(data.model)
      })
      .catch(() => {
        setStatusLabel('OFFLINE')
      })
  }

  const refreshContext = () => {
    if (DEMO_MODE) return

    setIsLoadingContext(true)
    fetch(`/api/context/${threadId}`)
      .then(response => response.json() as Promise<ContextResponse>)
      .then(data => {
        const calls = data.model_calls ?? []
        setModelCalls(calls)
        setSelectedCallId(current => (current && calls.some(call => call.call_id === current) ? current : null))
      })
      .finally(() => setIsLoadingContext(false))
  }

  useEffect(() => {
    refreshHealth()
    const timer = window.setInterval(refreshHealth, 5000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = '0px'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`
  }, [input])

  useEffect(() => {
    if (!messages.length || selectedCall) return
    if (isSending || messages[messages.length - 1]?.role === 'user') {
      conversationEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [messages, isSending, selectedCall])

  useEffect(() => {
    const stopMiddleScroll = () => {
      scrollSessionRef.current = null
      document.body.classList.remove('middle-scroll-active')
    }

    const handleMouseDown = (event: MouseEvent) => {
      if (event.button !== 1) return
      const target = event.target as HTMLElement | null
      const container = target?.closest<HTMLElement>('[data-middle-scroll="true"]')
      if (!container) return
      const canScrollY = container.scrollHeight > container.clientHeight
      const canScrollX = container.scrollWidth > container.clientWidth
      if (!canScrollX && !canScrollY) return

      event.preventDefault()
      scrollSessionRef.current = {
        container,
        startX: event.clientX,
        startY: event.clientY,
        startScrollTop: container.scrollTop,
        startScrollLeft: container.scrollLeft,
        canScrollX,
        canScrollY,
      }
      document.body.classList.add('middle-scroll-active')
    }

    const handleMouseMove = (event: MouseEvent) => {
      const session = scrollSessionRef.current
      if (!session) return
      event.preventDefault()
      const deltaX = event.clientX - session.startX
      const deltaY = event.clientY - session.startY
      if (session.canScrollY) session.container.scrollTop = session.startScrollTop - deltaY
      if (session.canScrollX) session.container.scrollLeft = session.startScrollLeft - deltaX
    }

    const preventAuxClick = (event: MouseEvent) => {
      if (event.button === 1 && scrollSessionRef.current) event.preventDefault()
    }

    window.addEventListener('mousedown', handleMouseDown, { passive: false })
    window.addEventListener('mousemove', handleMouseMove, { passive: false })
    window.addEventListener('mouseup', stopMiddleScroll)
    window.addEventListener('blur', stopMiddleScroll)
    window.addEventListener('keydown', stopMiddleScroll)
    window.addEventListener('auxclick', preventAuxClick, { passive: false })

    return () => {
      stopMiddleScroll()
      window.removeEventListener('mousedown', handleMouseDown)
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', stopMiddleScroll)
      window.removeEventListener('blur', stopMiddleScroll)
      window.removeEventListener('keydown', stopMiddleScroll)
      window.removeEventListener('auxclick', preventAuxClick)
    }
  }, [])

  const appendMessage = (role: Role, content: string) => {
    setMessages(current => [
      ...current,
      {
        id: `${role}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        role,
        content,
        createdAt: new Date().toISOString(),
      },
    ])
  }

  const handleNewChat = () => {
    if (DEMO_MODE) {
      setMessages([])
      setInput('')
      setThreadId(createThreadId())
      setModelCalls([])
      setSelectedCallId(null)
      setShowThinkModal(false)
      return
    }

    fetch(`/api/conversations/${threadId}`, { method: 'DELETE' }).finally(() => {
      setMessages([])
      setInput('')
      setThreadId(createThreadId())
      setModelCalls([])
      setSelectedCallId(null)
      setShowThinkModal(false)
    })
  }

  const sendMessage = async (content: string) => {
    const trimmed = content.trim()
    if (!trimmed || isSending) return

    appendMessage('user', trimmed)
    setInput('')
    setIsSending(true)
    setModelCalls([])
    setSelectedCallId(null)

    const assistantMsgId = `assistant_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    setMessages(current => [
      ...current,
      {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
        toolCalls: [],
      },
    ])

    try {
      if (DEMO_MODE) {
        const calls = buildDemoModelCalls(threadId, trimmed)
        const answer = buildDemoAnswer()
        let fullContent = ''

        await sleep(260)
        setModelCalls([calls[0]])
        setMessages(current =>
          current.map(message =>
            message.id === assistantMsgId
              ? {
                  ...message,
                  toolCalls: [
                    {
                      id: `demo_tc_${Date.now()}`,
                      tool: 'hybrid_search',
                      args: 'mmWave radar interference mitigation',
                      status: 'calling',
                    },
                  ],
                }
              : message,
          ),
        )

        await sleep(520)
        setMessages(current =>
          current.map(message => {
            if (message.id !== assistantMsgId) return message
            const toolCalls = message.toolCalls ?? []
            return {
              ...message,
              toolCalls: toolCalls.map((toolCall, index) =>
                index === 0
                  ? {
                      ...toolCall,
                      status: 'completed',
                      resultPreview: '返回 3 类演示论文证据：FMCW 互扰、深度学习抑制、时频域重构。',
                    }
                  : toolCall,
              ),
            }
          }),
        )
        setModelCalls([calls[0], calls[1]])

        await sleep(420)
        setModelCalls(calls)

        for (const char of answer) {
          fullContent += char
          setMessages(current =>
            current.map(message => (message.id === assistantMsgId ? { ...message, content: fullContent } : message)),
          )
          await sleep(char === '\n' ? 45 : 10)
        }

        return
      }

      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, thread_id: threadId }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`stream request failed: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let fullContent = ''
      let currentThreadId = threadId
      let buffer = ''

      while (reader) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue

          try {
            const data = JSON.parse(line.slice(6))

            if (data.type === 'text') {
              fullContent += data.content
              setMessages(current =>
                current.map(message =>
                  message.id === assistantMsgId ? { ...message, content: fullContent } : message,
                ),
              )
            } else if (data.type === 'tool_call') {
              const toolCall: ToolCall = {
                id: `tc_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
                tool: data.tool,
                args: data.args,
                status: 'calling',
              }
              setMessages(current =>
                current.map(message =>
                  message.id === assistantMsgId
                    ? { ...message, toolCalls: [...(message.toolCalls || []), toolCall] }
                    : message,
                ),
              )
            } else if (data.type === 'tool_result') {
              setMessages(current =>
                current.map(message => {
                  if (message.id !== assistantMsgId) return message
                  const toolCalls = message.toolCalls || []
                  if (toolCalls.length === 0) return message
                  const updatedToolCalls = [...toolCalls]
                  const lastIndex = updatedToolCalls.length - 1
                  updatedToolCalls[lastIndex] = {
                    ...updatedToolCalls[lastIndex],
                    status: 'completed',
                    resultPreview: data.preview,
                  }
                  return { ...message, toolCalls: updatedToolCalls }
                }),
              )
            } else if (data.type === 'model_call') {
              const nextCall = data.model_call as BackendModelCallRecord
              setModelCalls(current => {
                const filtered = current.filter(call => call.call_id !== nextCall.call_id)
                return [...filtered, nextCall].sort((a, b) => a.call_index - b.call_index)
              })
            } else if (data.type === 'error') {
              const errorText = data.content || '当前请求执行失败'
              setMessages(current =>
                current.map(message =>
                  message.id === assistantMsgId && !message.content
                    ? { ...message, content: `请求失败：${errorText}` }
                    : message,
                ),
              )
            } else if (data.type === 'done') {
              currentThreadId = data.thread_id || currentThreadId
            }
          } catch {
            // Ignore malformed SSE fragments.
          }
        }
      }

      setThreadId(currentThreadId)
      window.setTimeout(refreshContext, 220)
    } catch (error) {
      console.error('Stream error:', error)
      refreshHealth()
      setMessages(current =>
        current.map(message =>
          message.id === assistantMsgId
            ? { ...message, content: '当前没有拿到可用响应，请检查后端和模型服务是否正常。' }
            : message,
        ),
      )
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

  return (
    <div className="console-shell">
      <div className="star-field" />
      <div className="aurora-field" />
      <div className="orbit-ring ring-one" />
      <div className="orbit-ring ring-two" />

      <aside className="side-rail" data-middle-scroll="true">
        <section className="brand-panel">
          <div className="brand-mark">AX</div>
          <div>
            <h1>Agent Console</h1>
            <p>真实运行链路，按每一次模型调用观察上下文、状态和输出。</p>
          </div>
        </section>

        <section className="rail-actions">
          <button className="primary-action" onClick={handleNewChat} type="button">
            新建会话
          </button>
          <button className="ghost-action" onClick={() => setShowContextPanel(true)} type="button">
            深层 Context
          </button>
        </section>

        <section className="rail-card status-card">
          <span>Runtime</span>
          <strong>{statusLabel}</strong>
          <code>{activeModel}</code>
        </section>

        <section className="rail-card compact">
          <span>Current Turn</span>
          <div className="rail-metrics">
            <strong>
              {modelCalls.length}
              <small>LLM Calls</small>
            </strong>
            <strong>
              {toolCallCount}
              <small>Tools</small>
            </strong>
            <strong>
              {latestPhase}
              <small>Phase</small>
            </strong>
          </div>
        </section>

        <section className="rail-card scanner-card">
          <span>Signal</span>
          <div className="scanner-lines">
            <i />
            <i />
            <i />
            <i />
          </div>
          <p>{isSending ? 'streaming model output' : 'standing by'}</p>
        </section>

        <section className="rail-card">
          <span>Launch Pads</span>
          <div className="suggestion-list">
            {suggestions.map(item => (
              <button key={item} onClick={() => setInput(item)} type="button">
                {item}
              </button>
            ))}
          </div>
        </section>
      </aside>

      <main className="console-main">
        <header className="top-strip">
          <div>
            <span className="kicker">Live Agent Runtime</span>
            <h2>Agent Debug Workbench</h2>
          </div>
          <div className="top-meta">
            <div>
              <span>Selected</span>
              <strong>{selectedCall ? displayCallId(selectedCall) : '未展开'}</strong>
            </div>
            <div>
              <span>Thread</span>
              <code>{threadId}</code>
            </div>
          </div>
        </header>

        <section className="console-layout">
          <section className="main-story" data-middle-scroll="true">
            {selectedCall ? (
              <CallFocus
                call={selectedCall}
                onClose={() => setSelectedCallId(null)}
                onShowThink={() => setShowThinkModal(true)}
              />
            ) : (
              <>
                <div className="section-title">
                  <span>Conversation</span>
                  <h3>对话</h3>
                </div>
                <ConversationHud
                  modelCalls={modelCalls}
                  toolCallCount={toolCallCount}
                  latestPhase={latestPhase}
                  isSending={isSending}
                />
                {messages.length === 0 ? (
                  <div className="empty-transmission">
                    <span />
                    <strong>等待第一条用户请求</strong>
                    <p>发送消息后，右侧会显示后端记录的真实模型调用。</p>
                  </div>
                ) : (
                  <div className="message-list">
                    {messages.map(message => (
                      <article className={`message-card ${message.role}`} key={message.id}>
                        <div className="avatar">{message.role === 'user' ? '你' : 'AI'}</div>
                        <div className="message-body">
                          <div className="message-meta">
                            <strong>{message.role === 'user' ? '用户' : 'Agent'}</strong>
                          </div>
                          {message.role === 'assistant' && Boolean(message.toolCalls?.length) && (
                            <div className="tool-strip">
                              {message.toolCalls?.map(toolCall => (
                                <div className={`tool-pill ${toolCall.status}`} key={toolCall.id}>
                                  <span>{toolCall.status === 'completed' ? '完成' : '调用中'}</span>
                                  <strong>{toolCall.tool}</strong>
                                </div>
                              ))}
                            </div>
                          )}
                          {message.role === 'assistant' ? (
                            <div className="markdown-body">
                              <ReactMarkdown>{message.content}</ReactMarkdown>
                            </div>
                          ) : (
                            <p className="plain-message">{message.content}</p>
                          )}
                        </div>
                      </article>
                    ))}
                    {isSending && (
                      <article className="message-card assistant scanning">
                        <div className="avatar">AI</div>
                        <div className="message-body">
                          <div className="message-meta">
                            <strong>Agent</strong>
                          </div>
                          <div className="typing-dots">
                            <span />
                            <span />
                            <span />
                          </div>
                        </div>
                      </article>
                    )}
                    <div ref={conversationEndRef} />
                  </div>
                )}
              </>
            )}
          </section>

          <aside className="call-inspector">
            <section className="inspector-panel call-panel">
              <div className="inspector-head">
                <span>Model Calls</span>
                <strong>{modelCalls.length || 0} calls</strong>
              </div>
              <div className="call-id-list" data-middle-scroll="true">
                {modelCalls.length === 0 ? (
                  <div className="call-empty">no call captured</div>
                ) : (
                  modelCalls.map(call => {
                    const active = selectedCallId === call.call_id
                    return (
                      <button
                        className={active ? 'active' : ''}
                        key={call.call_id}
                        onClick={() => setSelectedCallId(current => (current === call.call_id ? null : call.call_id))}
                        type="button"
                      >
                        <code>{displayCallId(call)}</code>
                        <span>{active ? 'projected' : call.phase}</span>
                      </button>
                    )
                  })
                )}
              </div>
              <div className="projection-note">
                <strong>{selectedCall ? `${displayCallId(selectedCall)} 已投射到中间窗口` : '点击 call_id 在中间展开详情'}</strong>
              </div>
            </section>

            <section className="inspector-panel">
              <div className="inspector-head slim">
                <span>State Snapshot</span>
                <strong>{selectedCall ? 'linked' : 'collapsed'}</strong>
              </div>
              <div className="panel-placeholder">
                {selectedCall ? stateSummary(selectedCall.state_snapshot) : '选择一个 call 后，中间窗口会展示完整状态快照。'}
              </div>
            </section>

            <section className="inspector-panel">
              <div className="inspector-head slim">
                <span>Model Output</span>
                <strong>{getOutputType(selectedCall)}</strong>
              </div>
              <div className="panel-placeholder">
                {selectedCall
                  ? getToolName(selectedCall)
                    ? `${getOutputType(selectedCall)} · ${getToolName(selectedCall)}`
                    : getOutputSummary(selectedCall)
                  : '选择一个 call 后，中间窗口会展示模型输出与 raw think 入口。'}
              </div>
            </section>
          </aside>
        </section>

        <footer className="composer-wrap">
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              className="composer-input"
              disabled={isSending}
              onChange={event => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入消息，Enter 发送，Shift + Enter 换行"
              rows={1}
              value={input}
            />
            <button className="send-button" disabled={!input.trim() || isSending} type="submit">
              发送
            </button>
          </form>
        </footer>
      </main>

      {showThinkModal && selectedCall?.raw_think && (
        <aside className="modal-backdrop" role="dialog" aria-modal="true">
          <section className="think-modal" data-middle-scroll="true">
            <div className="drawer-header">
              <div>
                <span className="kicker">Raw Think</span>
                <h3>{displayCallId(selectedCall)}</h3>
              </div>
              <button onClick={() => setShowThinkModal(false)} type="button">
                关闭
              </button>
            </div>
            <pre>{selectedCall.raw_think}</pre>
          </section>
        </aside>
      )}

      {showContextPanel && (
        <aside className="context-drawer" data-middle-scroll="true">
          <div className="drawer-header">
            <div>
              <span className="kicker">Deep Context</span>
              <h3>当前线程模型调用</h3>
            </div>
            <div className="drawer-actions">
              <button onClick={refreshContext} type="button">
                {isLoadingContext ? '刷新中' : '刷新'}
              </button>
              <button onClick={() => setShowContextPanel(false)} type="button">
                关闭
              </button>
            </div>
          </div>
          <article className="drawer-card">
            <strong>model_calls</strong>
            <pre>{formatDisplayContent(modelCalls)}</pre>
          </article>
        </aside>
      )}
    </div>
  )
}

export default App
