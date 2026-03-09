import { useState, useRef, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { MessageInput } from './components/MessageInput';
import { Menu } from 'lucide-react';

// Interfaces (you can move to separate file later)
interface Source {
  id: string;
  name: string;
  url: string;
  snippet?: string;
}

interface Relate {
  question: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  relatedQuestions?: Relate[];
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm an AI assistant. How can I help you today?",
      sources: [],
      relatedQuestions: [],
    },
  ]);
  const [currentChatId, setCurrentChatId] = useState('1');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [sources, setSources] = useState<Source[]>([]);
  const [relatedQuestions, setRelatedQuestions] = useState<Relate[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom(); // 
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Reset for new query
    setSources([]);
    setRelatedQuestions([]);

    // Add user message
    const userId = Date.now().toString();
    setMessages(prev => [...prev, {
      id: userId,
      role: 'user',
      content: content.trim(),
      sources: [],
      relatedQuestions: [],
    }]);

    setIsLoading(true);

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: content.trim(),
          search_uuid: crypto.randomUUID(),
          generate_related_questions: true,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      // Add placeholder assistant message
      const assistantId = (Date.now() + 1).toString();
      setMessages(prev => [...prev, {
        id: assistantId,
        role: 'assistant',
        content: '',
        sources: [],
        relatedQuestions: [],
      }]);

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No stream');

      let buffer = '';
      let answer = '';
      let started = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += new TextDecoder().decode(value);

        if (!started) {
          const markerPos = buffer.indexOf('__LLM_RESPONSE__');
          if (markerPos !== -1) {
            // Parse sources (JSON array before marker)
            const sourcesJson = buffer.slice(0, markerPos).trim();
            buffer = buffer.slice(markerPos + 16).trimStart();

            try {
              const parsedSources = JSON.parse(sourcesJson);
              setSources(parsedSources);
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1].sources = parsedSources;
                return updated;
              });
            } catch (e) {
              console.error("Parse sources error:", e);
            }

            started = true;
          } else {
            buffer = '';
            continue;
          }
        }

        const relatedPos = buffer.indexOf('__RELATED_QUESTIONS__');
        if (relatedPos !== -1) {
          answer += buffer.slice(0, relatedPos);
          const relatesJson = buffer.slice(relatedPos + 21).trim();

          try {
            const parsedRelates = JSON.parse(relatesJson);
            setRelatedQuestions(parsedRelates);
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1].relatedQuestions = parsedRelates;
              return updated;
            });
          } catch (e) {
            console.error("Parse related error:", e);
          }

          buffer = '';
        } else {
          answer += buffer;
          buffer = '';
        }

        // Live update answer
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1].content = answer;
          return updated;
        });
      }

      // Final flush
      if (buffer && started) {
        answer += buffer.trim();
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1].content = answer;
          return updated;
        });
      }
    } catch (err) {
      if ((err as DOMException).name === 'AbortError') {
        setMessages(prev => [...prev, {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content: '[Generation stopped]',
        }]);
      } else {
        console.error('Chat error:', err);
        setMessages(prev => [...prev, {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content: 'Sorry, something went wrong. Please try again.',
        }]);
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStopGeneration = () => {
    abortControllerRef.current?.abort();
  };

  const handleNewChat = () => {
    setMessages([
      {
        id: Date.now().toString(),
        role: 'assistant',
        content: "Hello! I'm an AI assistant. How can I help you today?",
        sources: [],
        relatedQuestions: [],
      },
    ]);
    setCurrentChatId(Date.now().toString());
    setSources([]);
    setRelatedQuestions([]);
  };

  const handleSelectChat = (chatId: string) => {
    setCurrentChatId(chatId);
    setMessages([
      {
        id: '1',
        role: 'assistant',
        content: `This is the chat history for conversation ${chatId}. In a real implementation, messages would be loaded from storage.`,
        sources: [],
        relatedQuestions: [],
      },
    ]);
    setSources([]);
    setRelatedQuestions([]);
  };

  return (
    <div className="flex h-screen bg-amber-50 text-gray-900 overflow-hidden">
      <Sidebar
        onNewChat={handleNewChat}
        currentChatId={currentChatId}
        onSelectChat={handleSelectChat}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
      />

      <div
        className={`flex-1 flex flex-col relative transition-all duration-300 ${
          isSidebarOpen ? 'ml-64' : 'ml-0'
        }`}
      >
        <div className="border-b border-amber-200 bg-white/95 backdrop-blur-sm">
          <div className="w-full px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                className="p-2 rounded-lg hover:bg-amber-50 transition-colors"
                title={isSidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
              >
                <Menu size={20} />
              </button>
              <img
                src="/assets/logo.png"
                alt="ECEasy"
                className="h-12"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                  const span = document.createElement('span');
                  span.textContent = 'ECEasy';
                  span.className = 'text-xl font-bold text-blue-700';
                  e.target.parentNode?.appendChild(span);
                }}
              />
            </div>

            <div className="absolute left-1/2 transform -translate-x-1/2">
              <h1 className="text-2xl font-bold">
                <span style={{ color: '#1e3a8a' }}>EC</span>
                <span style={{ color: '#3b82f6' }}>Easy</span>
              </h1>
            </div>

            <div className="w-[88px]"></div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto bg-amber-50">
          <div className="min-h-full flex flex-col">
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                role={message.role}
                content={message.content}
                sources={message.sources}
                relatedQuestions={message.relatedQuestions}
                onQuestionClick={handleSendMessage}
              />
            ))}
            {isLoading && (
              <div className="flex gap-4 px-4 py-6 bg-white">
                <div className="max-w-4xl mx-auto w-full flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full flex items-center justify-center bg-gradient-to-br from-amber-500 to-yellow-500">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0 pt-1">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <MessageInput
          onSendMessage={handleSendMessage}
          disabled={isLoading}
          isLoading={isLoading}
          onStopGeneration={handleStopGeneration}
        />
      </div>
    </div>
  );
}