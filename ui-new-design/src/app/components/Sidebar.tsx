import { Plus, MessageSquare, Menu, X } from 'lucide-react';

interface Chat {
  id: string;
  title: string;
  timestamp: string;
}

interface SidebarProps {
  onNewChat: () => void;
  currentChatId: string;
  onSelectChat: (chatId: string) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

export function Sidebar({ onNewChat, currentChatId, onSelectChat, isOpen, setIsOpen }: SidebarProps) {
  // Mock chat history
  const chats: Chat[] = [
    { id: '1', title: 'React best practices', timestamp: 'Today' },
    { id: '2', title: 'AI model comparison', timestamp: 'Today' },
    { id: '3', title: 'CSS grid layout help', timestamp: 'Yesterday' },
    { id: '4', title: 'Python data structures', timestamp: 'Yesterday' },
    { id: '5', title: 'Machine learning basics', timestamp: 'Feb 12' },
    { id: '6', title: 'API design patterns', timestamp: 'Feb 11' },
    { id: '7', title: 'Database optimization', timestamp: 'Feb 10' },
  ];

  return (
    <>
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 bg-amber-50 border-r border-amber-200 flex flex-col transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* New Chat Button */}
        <div className="p-3 border-b border-amber-200">
          <button
            onClick={onNewChat}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-amber-100 hover:bg-amber-200 transition-colors text-gray-900"
          >
            <Plus size={18} />
            <span>New chat</span>
          </button>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {chats.map((chat) => (
            <button
              key={chat.id}
              onClick={() => {
                onSelectChat(chat.id);
                if (window.innerWidth < 1024) setIsOpen(false);
              }}
              className={`w-full flex items-start gap-3 px-3 py-3 rounded-lg transition-colors text-left ${
                currentChatId === chat.id
                  ? 'bg-amber-100 text-gray-900'
                  : 'text-gray-600 hover:bg-amber-100/50 hover:text-gray-900'
              }`}
            >
              <MessageSquare size={18} className="mt-0.5 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="truncate text-sm">{chat.title}</div>
                <div className="text-xs text-gray-400 mt-0.5">{chat.timestamp}</div>
              </div>
            </button>
          ))}
        </div>

        {/* User Section */}
        <div className="p-4 border-t border-amber-200">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center text-sm text-white">
              U
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm truncate text-gray-900">User Account</div>
              <div className="text-xs text-gray-500">Free Plan</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}