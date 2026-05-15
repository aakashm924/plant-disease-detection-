import { useEffect, useRef } from 'react';
import { X, Leaf, Send, MessageSquare } from 'lucide-react';

interface Message {
  text: string;
  sender: 'user' | 'bot';
  loading?: boolean;
}

interface ChatWidgetProps {
  isOpen: boolean;
  onClose: () => void;
  onOpen: () => void;
  messages: Message[];
  query: string;
  setQuery: (q: string) => void;
  askBot: (text?: string) => void;
}

const QUICK_QUESTIONS = [
  "How to treat powdery mildew?",
  "Signs of root rot?",
  "Best organic fungicide?",
  "Why are my leaves yellowing?",
];

export default function ChatWidget({
  isOpen, onClose, onOpen, messages, query, setQuery, askBot
}: ChatWidgetProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 150);
  }, [isOpen]);

  if (!isOpen) {
    return (
      <button
        type="button"
        onClick={onOpen}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-emerald-600 text-white shadow-lg shadow-emerald-900/30 hover:bg-emerald-500 transition-all hover:scale-105 flex items-center justify-center z-40"
        aria-label="Open PlantDoc chat"
      >
        <MessageSquare size={22} />
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 w-80 h-[480px] flex flex-col rounded-2xl bg-white shadow-2xl shadow-black/20 z-50 border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-700 to-emerald-600 px-4 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
            <Leaf size={16} className="text-white" />
          </div>
          <div>
            <div className="text-white font-bold text-sm">PlantDoc AI</div>
            <div className="text-emerald-200 text-xs">Powered by Claude</div>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-white/70 hover:text-white transition-colors p-1"
          aria-label="Close chat"
        >
          <X size={18} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.length === 0 && (
          <div className="space-y-3">
            <div className="bg-white rounded-xl rounded-tl-sm p-3 border border-gray-100 shadow-sm max-w-[88%]">
              <p className="text-sm text-gray-700 leading-relaxed">
                👋 Hi! I'm <strong>PlantDoc AI</strong>, your plant health expert. Ask me anything about plant diseases, treatments, or care tips!
              </p>
            </div>
            <p className="text-xs text-gray-400 text-center">Quick questions:</p>
            <div className="space-y-2">
              {QUICK_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => askBot(q)}
                  className="w-full text-left text-xs bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 rounded-lg px-3 py-2 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={`msg-${i}`}
            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.sender === 'bot' && (
              <div className="w-6 h-6 bg-emerald-100 rounded-full flex items-center justify-center mr-1.5 flex-shrink-0 mt-auto mb-0.5">
                <Leaf size={12} className="text-emerald-600" />
              </div>
            )}
            <div
              className={`max-w-[78%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                msg.sender === 'user'
                  ? 'bg-emerald-600 text-white rounded-tr-sm'
                  : 'bg-white text-gray-700 rounded-tl-sm border border-gray-100 shadow-sm'
              }`}
            >
              {msg.loading ? (
                <span className="flex gap-1 items-center py-0.5">
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </span>
              ) : (
                msg.text
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 bg-white border-t border-gray-100">
        <div className="flex gap-2 items-center bg-gray-50 rounded-xl border border-gray-200 px-3 py-2">
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') askBot(); }}
            className="flex-1 bg-transparent text-sm text-gray-700 placeholder-gray-400 outline-none min-w-0"
            placeholder="Ask about plant diseases..."
          />
          <button
            type="button"
            onClick={() => askBot()}
            disabled={!query.trim()}
            className="w-7 h-7 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-40 text-white rounded-lg flex items-center justify-center transition-colors flex-shrink-0"
            aria-label="Send"
          >
            <Send size={13} />
          </button>
        </div>
      </div>
    </div>
  );
}
