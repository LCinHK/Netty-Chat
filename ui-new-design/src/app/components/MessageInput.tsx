import { Send, Paperclip, X, Square } from 'lucide-react';
import { useState, KeyboardEvent, useRef } from 'react';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  isLoading?: boolean;
  onStopGeneration?: () => void;
}

export function MessageInput({ onSendMessage, disabled, isLoading, onStopGeneration }: MessageInputProps) {
  const [message, setMessage] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    if ((message.trim() || uploadedFiles.length > 0) && !disabled && !isLoading) {
      onSendMessage(message.trim());
      setMessage('');
      setUploadedFiles([]);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const newFiles = Array.from(files);
      setUploadedFiles((prev) => [...prev, ...newFiles]);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="border-t border-amber-200 bg-white/95 backdrop-blur-sm">
      <div className="max-w-4xl mx-auto px-4 py-4">
        {/* Uploaded Files Display */}
        {uploadedFiles.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {uploadedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center gap-2 bg-orange-50 rounded-lg px-3 py-2 text-sm"
              >
                <Paperclip size={14} className="text-orange-600" />
                <span className="text-gray-700 max-w-[200px] truncate">
                  {file.name}
                </span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-orange-500 hover:text-orange-700 transition-colors"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="relative flex items-end gap-2 bg-amber-100 rounded-2xl p-2">
          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.doc,.docx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />

          {/* Upload Button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || isLoading}
            className="flex-shrink-0 p-2 rounded-lg hover:bg-amber-200 transition-colors text-amber-700 hover:text-amber-900 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Upload files"
          >
            <Paperclip size={18} />
          </button>

          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled || isLoading}
            placeholder="Message ECEasy..."
            rows={1}
            className="flex-1 bg-transparent text-gray-900 placeholder-gray-400 resize-none outline-none px-3 py-2 max-h-[200px] min-h-[24px]"
            style={{
              height: 'auto',
              maxHeight: '200px',
            }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = 'auto';
              target.style.height = target.scrollHeight + 'px';
            }}
          />
          
          {/* Send or Stop Button */}
          {isLoading ? (
            <button
              onClick={onStopGeneration}
              className="flex-shrink-0 p-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors"
              title="Stop generation"
            >
              <Square size={18} fill="currentColor" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={(!message.trim() && uploadedFiles.length === 0) || disabled || isLoading}
              className={`flex-shrink-0 p-2 rounded-lg transition-colors ${
                (message.trim() || uploadedFiles.length > 0) && !disabled && !isLoading
                  ? 'bg-amber-600 text-white hover:bg-amber-700'
                  : 'bg-amber-300 text-amber-500 cursor-not-allowed'
              }`}
            >
              <Send size={18} />
            </button>
          )}
        </div>
        <div className="text-xs text-gray-500 text-center mt-2">
          Press Enter to send, Shift + Enter for new line
        </div>
      </div>
    </div>
  );
}