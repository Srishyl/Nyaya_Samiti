import "../index.css";
import { Link } from "react-router-dom";
import Footer from "../components/Footer";
import FloatingChatbot from "../components/FloatingChatbot";

function Dashboard() {
  return (
    <div className="min-h-screen bg-[#F3ECDA]/80 text-[#2b1d14]">
      <header className="border-b border-[#c4ac95]/40 bg-[#94553D]">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-full bg-[#C4AC95]"></div>
              <span className="text-xl font-semibold tracking-wide text-[#F3ECDA]">
                NYAYA SAMITI
              </span>
            </div>
            <nav className="flex items-center gap-4">
              <Link
                to="/"
                className="text-sm font-medium text-[#F3ECDA] hover: hover:scale-105 transition-all duration-300 px-4 py-2 rounded-md"
              >
                Home
              </Link>
              
             
            </nav>
          </div>
        </div>
      </header>

      <main className="bg-[#F3ECDA]/70 py-12">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-[#94553D] mb-4">
              Dashboard
            </h1>
            <p className="text-lg text-[#94553D]/80">
              Manage your legal cases and documents
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Quick Actions */}
            <div className="bg-white/90 rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold text-[#94553D] mb-4">
                Quick Actions
              </h2>
              <div className="space-y-3">
                <Link
                  to="/document-validation"
                  className="block w-full bg-[#94553D] text-white px-4 py-3 rounded-lg text-center hover:bg-[#7a3d2f] traopgitnsition-colors"
                >
                  Validate Document
                </Link>
                <Link
                  to="/validate-and-file"
                  className="block w-full bg-[#FFCBA4] text-[#94553D] px-4 py-3 rounded-lg text-center hover:bg-[#f7b983] transition-colors"
                >
                  File New Case
                </Link>
                <Link
                  to="/user-registration"
                  className="block w-full border border-[#94553D] text-[#94553D] px-4 py-3 rounded-lg text-center hover:bg-[#94553D] hover:text-white transition-colors"
                >
                  Register User
                </Link>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white/90 rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold text-[#94553D] mb-4">
                Recent Activity
              </h2>
              <div className="space-y-3">
                <div className="border-l-4 border-[#94553D] pl-4">
                  <p className="text-sm text-gray-600">Document validated</p>
                  <p className="text-xs text-gray-500">2 hours ago</p>
                </div>
                <div className="border-l-4 border-[#FFCBA4] pl-4">
                  <p className="text-sm text-gray-600">Case filed</p>
                  <p className="text-xs text-gray-500">1 day ago</p>
                </div>
                <div className="border-l-4 border-green-500 pl-4">
                  <p className="text-sm text-gray-600">User registered</p>
                  <p className="text-xs text-gray-500">3 days ago</p>
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-white/90 rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold text-[#94553D] mb-4">
                Statistics
              </h2>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Documents Validated</span>
                  <span className="font-semibold text-[#94553D]">12</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Cases Filed</span>
                  <span className="font-semibold text-[#94553D]">5</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Users Registered</span>
                  <span className="font-semibold text-[#94553D]">8</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Success Rate</span>
                  <span className="font-semibold text-green-600">95%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Features Overview */}
          <div className="mt-12">
            <h2 className="text-2xl font-bold text-[#94553D] mb-8 text-center">
              Platform Features
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white/90 rounded-lg shadow-lg p-6 text-center">
                <div className="w-12 h-12 bg-[#94553D] rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-[#94553D] mb-2">Document Validation</h3>
                <p className="text-sm text-gray-600">AI-powered document verification and authenticity checking</p>
              </div>

              <div className="bg-white/90 rounded-lg shadow-lg p-6 text-center">
                <div className="w-12 h-12 bg-[#FFCBA4] rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-[#94553D]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </div>
                <h3 className="font-semibold text-[#94553D] mb-2">Case Filing</h3>
                <p className="text-sm text-gray-600">Streamlined legal case filing and management system</p>
              </div>

              <div className="bg-white/90 rounded-lg shadow-lg p-6 text-center">
                <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-[#94553D] mb-2">User Management</h3>
                <p className="text-sm text-gray-600">Comprehensive user registration and family member management</p>
              </div>

              <div className="bg-white/90 rounded-lg shadow-lg p-6 text-center">
                <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-[#94553D] mb-2">Analytics</h3>
                <p className="text-sm text-gray-600">Real-time analytics and reporting dashboard</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
      <FloatingChatbot />
    </div>
  );
}

export default Dashboard;
