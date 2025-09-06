import '../index.css'
import { Link } from 'react-router-dom'

function Homepage() {
  return (
    <div className="min-h-screen bg-[#F3ECDA] text-[#94553D]">
      <header className="border-b border-[#c4ac95]/40 bg-[#94553D]">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-full bg-[#F3ECDA]"></div>
              <span className="text-xl font-semibold tracking-wide text-[#F3ECDA]">NYAYA SAMITI</span>
            </div>
            <nav className="flex items-center gap-4">
             
              <Link
                to="/profile"
                className="text-sm font-medium text-[#F3ECDA] hover: hover:scale-105 transition-all duration-300 px-4 py-2 rounded-md"
              >
                Profile
                </Link>
              <Link
                to="/dashboard"
                className="text-sm font-medium text-[#F3ECDA] hover: hover:scale-105 transition-all duration-300 px-4 py-2 rounded-md"
              >
                Dashboard
              </Link>
              
              <Link
                to="/dashboard"
                className="text-sm font-medium text-[#F3ECDA] hover: hover:scale-105 transition-all duration-300 px-4 py-2 rounded-md"
              >
                Sign In
              </Link>
             
            </nav>
          </div>
        </div>
      </header>

      <main className="bg-[#F3ECDA]">
        <section className="relative isolate">
          <div className="absolute inset-0 bg-[#F3ECDA] text-[#94553D]"></div>
          <div className="relative mx-auto grid max-w-7xl grid-cols-1 gap-12 px-4 py-24 sm:px-6 lg:grid-cols-12 lg:gap-16 lg:px-8">
            <div className="lg:col-span-7">
              <div className="inline-flex items-center rounded-full bg-[#F3ECDA]/15 px-3 py-1 text-xs text-[#F3ECDA] ring-1 ring-inset ring-[#F3ECDA]/25 backdrop-blur">
                From Detection to Direction
              </div>
              <h1 className="mt-6 text-4xl font-extrabold leading-tight tracking-tight text-[#94553D] sm:text-5xl lg:text-6xl">
                AI-Powered Legal Document Verification & Guidance
              </h1>
              <p className="mt-6 max-w-2xl text-base leading-7 text-[#94553D]/90 sm:text-lg">
                Making legal processes faster, transparent, and fraud-proof for citizens, government, and enterprises.
              </p>
              <div className="mt-10 flex flex-wrap items-center gap-4">
                <Link to="/document-validation" className="rounded-md bg-[#FFCBA4] px-5 py-3 text-sm font-semibold text-[#94553D] shadow-sm transition hover:bg-[#f7b983] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#FFCBA4]">
                  Validate Document
                </Link>
                
              </div>
            </div>

           
          </div>
        </section>
      </main>

      <footer className="border-t border-[#c4ac95]/40 bg-[#94553D]">
        <div className="mx-auto max-w-7xl px-4 py-8 text-sm text-[#F3ECDA] sm:px-6 lg:px-8">
          © {new Date().getFullYear()} NYAYA SAMITI. All rights reserved.
        </div>
      </footer>
    </div>
  )
}

export default Homepage


