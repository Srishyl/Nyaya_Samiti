import { useState, useRef } from 'react'

const GEMINI_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'

const SYSTEM_INSTRUCTION = `You are NYAYA SAMITI's legal assistant for India. Answer only questions related to:
- Legal documents (formats, validation, verification)
- Indian Constitution (Articles, Schedules, Fundamental Rights, DPSP)
- Indian laws (IPC, CrPC, Evidence Act, IT Act, Contract Act, etc.)
- Courts, procedures, compliance, and governance

Strictly refuse unrelated topics. Be concise, neutral, and cite relevant Acts/Articles when helpful.
FORMAT your entire response in Markdown with these sections in order and keep it concise:
## Title
## Summary
## Key Points
- bullet list
## Relevant Law/Citations
- bullet list with Act/Article/Section numbers
## Guidance
Short, practical next steps in 2-4 lines.
\nNote: Not legal advice.`

function isLegalQuery(text) {
  const t = (text || '').toLowerCase()
  const keywords = [
    'law', 'legal', 'constitution', 'ipc', 'crpc', 'evidence act', 'it act', 'contract act', 'article', 'fundamental rights', 'dpsp', 'court', 'supreme court', 'high court', 'bail', 'fir', 'writ', 'petition', 'stamp', 'notary', 'affidavit', 'aadhaar', 'pan', 'compliance', 'verification', 'document', 'act', 'section'
  ]
  return keywords.some(k => t.includes(k))
}

export default function LegalChatbot() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const listRef = useRef(null)

  async function handleSend(e) {
    e.preventDefault()
    setError('')
    const content = input.trim()
    if (!content) return

    if (!isLegalQuery(content)) {
      setMessages(prev => [...prev, { role: 'user', text: content }, { role: 'assistant', text: "I'm focused on Indian legal topics. Ask about legal documents, the Constitution, or Indian laws (IPC/CrPC/Evidence Act, etc.)." }])
      setInput('')
      queueMicrotask(() => listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' }))
      return
    }

    const apiKey = import.meta.env.VITE_GEMINI_API_KEY
    if (!apiKey) {
      setError('Missing Gemini API key. Set VITE_GEMINI_API_KEY in .env')
      return
    }

    const newMessages = [...messages, { role: 'user', text: content }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const contents = [
        { role: 'user', parts: [{ text: SYSTEM_INSTRUCTION }] },
        ...newMessages.slice(-10).map(m => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.text }] })),
      ]

      const res = await fetch(`${GEMINI_ENDPOINT}?key=${apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents, generationConfig: { temperature: 0.2, topP: 0.9, topK: 40, maxOutputTokens: 1024 } })
      })
      if (!res.ok) throw new Error(`API error ${res.status}`)
      const data = await res.json()
      const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || 'Sorry, I could not generate a response.'
      setMessages(prev => [...prev, { role: 'assistant', text }])
      queueMicrotask(() => listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' }))
    } catch (err) {
      setError('Request failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-lg border border-[#C4AC95]/60 bg-white/70 p-3 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-[#94553D]">Legal Assistant</h3>
        <span className="text-xs text-[#2b1d14]/60">Indian law & Constitution only</span>
      </div>
      <div ref={listRef} className="h-48 overflow-y-auto rounded-lg bg-[#F3ECDA] p-2">
        {messages.length === 0 && (
          <div className="text-xs text-[#2b1d14]/70">
            Ask about Articles of the Constitution, IPC sections, affidavits, notary, document verification, court procedures, etc.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`mb-2 flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] overflow-hidden rounded-lg px-2 py-1 text-xs ${m.role === 'user' ? 'bg-[#94553D] text-[#F3ECDA]' : 'bg-white/90 text-[#2b1d14] ring-1 ring-[#C4AC95]/60'}`}>
              {m.role === 'assistant' ? (
                <AssistantMarkdown text={m.text} />
              ) : (
                m.text
              )}
            </div>
          </div>
        ))}
      </div>
      <form onSubmit={handleSend} className="mt-2 flex items-center gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about Indian laws..."
          className="w-full rounded border border-[#C4AC95]/60 bg-white/80 px-2 py-1 text-xs text-[#2b1d14] placeholder-[#2b1d14]/50 focus:outline-none focus:ring-1 focus:ring-[#C4AC95]"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded bg-[#94553D] px-3 py-1 text-xs font-semibold text-[#F3ECDA] shadow-sm transition hover:bg-[#7a3e2a] disabled:opacity-50"
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
      </form>
      {error && <div className="mt-1 text-xs text-red-700">{error}</div>}
      <div className="mt-1 text-[10px] text-[#2b1d14]/60">Responses are informational only and not legal advice.</div>
    </div>
  )
}

function AssistantMarkdown({ text }) {
  // Minimal markdown renderer for headings and lists
  const lines = (text || '').split('\n')
  return (
    <div className="prose prose-sm prose-slate max-w-none prose-headings:text-[#2b1d14] prose-strong:text-[#2b1d14]">
      {lines.map((line, idx) => {
        if (line.startsWith('## ')) {
          return <h3 key={idx} className="mt-2 text-base font-semibold">{line.replace('## ', '')}</h3>
        }
        if (line.startsWith('- ')) {
          return <li key={idx} className="ml-5 list-disc">{line.replace('- ', '')}</li>
        }
        if (line.trim() === '') {
          return <div key={idx} className="h-2" />
        }
        return <p key={idx}>{line}</p>
      })}
    </div>
  )
}


