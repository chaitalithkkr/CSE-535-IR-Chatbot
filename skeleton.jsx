import React, { useState } from 'react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      id: messages.length,
      text: input,
      sender: 'user',
    };

    setMessages([...messages, userMessage]);
    setInput('');

    setTimeout(() => {
      const botMessage = {
        id: messages.length + 1,
        text: "Bot response placeholder",
        sender: 'bot',
      };
      setMessages(prev => [...prev, botMessage]);
    }, 500);
  };

  return (
    <div className="h-screen flex bg-white p-8 gap-6">
      {/* Main Chat Container */}
      <div className="relative border-2 border-blue-500 rounded-lg flex-1 p-4">
        {/* Dotted vertical divider */}
        <div className="absolute top-0 bottom-0 left-1/2 border-l-2 border-dotted border-blue-500" />

        {/* Headers */}
        <div className="flex text-sm mb-4">
          <div className="flex-1">Bot side</div>
          <div className="flex-1 pl-4">User side</div>
        </div>

        {/* Messages Container */}
        <div className="space-y-4 mb-16">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`w-[45%] h-12 rounded ${
                  message.sender === 'user'
                    ? 'bg-[#D5E6F9]'
                    : 'bg-[#E6E6E6]'
                }`}
              >
                <div className="p-3">{message.text}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Input Area with Form */}
        <div className="absolute bottom-4 left-4 right-4">
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="w-full h-12 rounded-full border-2 border-blue-500 px-4"
              placeholder="Type your message..."
            />
          </form>
        </div>
      </div>

      {/* Topics Box */}
      <div className="w-48 border-2 border-blue-500 rounded-lg p-4">
        <div className="font-medium mb-4">Topics</div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-blue-500 rounded" />
            <span>Topic 1</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-blue-500 rounded" />
            <span>Topic 2</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-blue-500 rounded" />
            <span>Topic 3</span>
          </div>
          <div className="mt-4 pt-2 border-t-2 border-blue-500">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-500 rounded" />
              <span>All (Default)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;