import { useState } from 'react'
import LegalChatbot from './LegalChatbot'

export default function FloatingChatbot() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      {/* Floating Chat Icon */}
      <div className="fixed bottom-6 right-6 z-50">
        <div className="relative">
          {/* Notification dot */}
          <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full animate-pulse border-2 border-white"></div>
          
          <button
            onClick={() => setIsOpen(true)}
            className="bg-gradient-to-br from-[#94553D] to-[#7a3d2f] hover:from-[#7a3d2f] hover:to-[#5d2e1f] text-[#F3ECDA] rounded-full p-5 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-110 border-2 border-[#F3ECDA]/20 animate-bounce"
            aria-label="Open Legal Chatbot"
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              viewBox="0 0 24 24" 
              fill="currentColor" 
              className="w-8 h-8"
            >
              <path d="M12 2C6.48 2 2 6.48 2 12c0 1.54.36 2.98.97 4.29L1 23l6.71-1.97C9.02 21.64 10.46 22 12 22c5.52 0 10-4.48 10-10S17.52 2 12 2zm-1 15h-2v-2h2v2zm0-4h-2V9h2v4zm4 4h-2v-2h2v2zm0-4h-2V9h2v4z"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Modal Overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="relative bg-white rounded-2xl shadow-2xl max-w-md w-full max-h-[80vh] overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-[#94553D] text-[#F3ECDA]">
              <h3 className="text-lg font-semibold">NYAYA SAMITI – Legal Assistant</h3>
              <button
                onClick={() => setIsOpen(false)}
                className="text-[#F3ECDA] hover:text-white transition-colors"
                aria-label="Close chatbot"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
              </button>
            </div>
            
            {/* Chatbot Content */}
            <div className="p-4">
              <LegalChatbot />
            </div>
          </div>
        </div>
      )}
    </>
  )
}
