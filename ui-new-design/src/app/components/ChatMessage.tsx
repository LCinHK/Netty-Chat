import { Bot, User, ExternalLink, BookOpen, HelpCircle } from 'lucide-react';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{
    title: string;
    url: string;
    snippet?: string;
  }>;
  relatedQuestions?: string[];
  onQuestionClick?: (question: string) => void;
}

export function ChatMessage({ role, content, sources, relatedQuestions, onQuestionClick }: ChatMessageProps) {
  const isUser = role === 'user';

  return (
    <div
      className={`flex gap-4 px-4 py-6 ${
        isUser ? 'bg-amber-100' : 'bg-amber-50'
      }`}
    >
      <div className="max-w-4xl mx-auto w-full flex gap-4">
        {/* Avatar */}
        <div className="flex-shrink-0">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center ${
              isUser
                ? 'bg-gradient-to-br from-amber-500 to-orange-500 text-white'
                : 'bg-gradient-to-br from-amber-500 to-yellow-500 text-white'
            }`}
          >
            {isUser ? <User size={18} /> : <Bot size={18} />}
          </div>
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0 space-y-4">
          {/* Section 1: Response */}
          <div className="bg-white rounded-lg p-4 border border-amber-200">
            <div className="prose max-w-none">
              <p className="text-gray-900 leading-7 whitespace-pre-wrap m-0">{content}</p>
            </div>
          </div>

          {/* Section 2: Cited Sources */}
          {sources && sources.length > 0 && (
            <div className="bg-white rounded-lg p-4 border border-amber-200">
              <div className="flex items-center gap-2 mb-3">
                <BookOpen size={18} className="text-amber-600" />
                <h3 className="text-sm font-semibold text-gray-900 m-0">
                  Cited Sources ({sources.length})
                </h3>
              </div>
              <div className="space-y-2">
                {sources.map((source, index) => (
                  <a
                    key={index}
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block p-3 bg-amber-50 rounded-lg hover:bg-amber-100 transition-all group border border-amber-100 hover:border-amber-200"
                  >
                    <div className="flex items-start gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-amber-200 text-amber-800 text-xs font-medium">
                            {index + 1}
                          </span>
                          <span className="text-sm font-medium text-gray-900 group-hover:text-amber-700 transition-colors">
                            {source.title}
                          </span>
                          <ExternalLink size={14} className="text-gray-400 group-hover:text-amber-600 flex-shrink-0" />
                        </div>
                        {source.snippet && (
                          <p className="text-xs text-gray-600 mt-2 ml-7 line-clamp-2 m-0">
                            {source.snippet}
                          </p>
                        )}
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* Section 3: Related Questions */}
          {relatedQuestions && relatedQuestions.length > 0 && (
            <div className="bg-white rounded-lg p-4 border border-amber-200">
              <div className="flex items-center gap-2 mb-3">
                <HelpCircle size={18} className="text-amber-600" />
                <h3 className="text-sm font-semibold text-gray-900 m-0">
                  Related Questions
                </h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {relatedQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => onQuestionClick?.(question)}
                    className="text-left p-3 bg-amber-50 rounded-lg hover:bg-amber-100 transition-all text-sm text-gray-700 hover:text-amber-800 border border-amber-100 hover:border-amber-200 hover:shadow-sm"
                  >
                    <div className="flex items-start gap-2">
                      <span className="text-amber-600 mt-0.5 flex-shrink-0">→</span>
                      <span className="flex-1">{question}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
