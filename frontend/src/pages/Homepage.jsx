import '../index.css'

function Homepage() {
  return (
    <div className="min-h-screen bg-[#F3ECDA] text-[#2b1d14]">
      <header className="border-b border-[#c4ac95]/40 bg-[#94553D]">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-full bg-[#C4AC95]"></div>
              <span className="text-xl font-semibold tracking-wide text-[#F3ECDA]">NYAYA SAMITI</span>
            </div>
          </div>
        </div>
      </header>

      <main className="bg-[#F3ECDA]">
        <section className="relative isolate">
          <div className="absolute inset-0 bg-gradient-to-b from-[#94553D] to-[#C4AC95]"></div>
          <div className="relative mx-auto grid max-w-7xl grid-cols-1 gap-12 px-4 py-24 sm:px-6 lg:grid-cols-12 lg:gap-16 lg:px-8">
            <div className="lg:col-span-7">
              <div className="inline-flex items-center rounded-full bg-[#F3ECDA]/15 px-3 py-1 text-xs text-[#F3ECDA] ring-1 ring-inset ring-[#F3ECDA]/25 backdrop-blur">
                From Detection to Direction
              </div>
              <h1 className="mt-6 text-4xl font-extrabold leading-tight tracking-tight text-[#F3ECDA] sm:text-5xl lg:text-6xl">
                AI-Powered Legal Document Verification & Guidance
              </h1>
              <p className="mt-6 max-w-2xl text-base leading-7 text-[#F3ECDA]/90 sm:text-lg">
                Making legal processes faster, transparent, and fraud-proof for citizens, government, and enterprises.
              </p>
              <div className="mt-10 flex flex-wrap items-center gap-4">
                <a href="#" className="rounded-md bg-[#FFCBA4] px-5 py-3 text-sm font-semibold text-[#94553D] shadow-sm transition hover:bg-[#f7b983] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#FFCBA4]">
                  Get Started
                </a>
                <a href="#" className="rounded-md px-5 py-3 text-sm font-semibold text-[#F3ECDA] ring-1 ring-inset ring-[#F3ECDA]/40 transition hover:bg-[#F3ECDA]/10">
                  Learn More
                </a>
              </div>
            </div>

            <div className="lg:col-span-5">
              <div className="rounded-xl border border-[#F3ECDA]/20 bg-[#F3ECDA]/10 p-6 backdrop-blur">
                <div className="grid grid-cols-2 gap-4">
                  <div className="rounded-lg bg-[#F3ECDA]/15 p-4 text-[#F3ECDA] ring-1 ring-inset ring-[#F3ECDA]/20">
                    Trusted Documents
                  </div>
                  <div className="rounded-lg bg-[#F3ECDA]/15 p-4 text-[#F3ECDA] ring-1 ring-inset ring-[#F3ECDA]/20">
                    Fraud Detection
                  </div>
                  <div className="rounded-lg bg-[#F3ECDA]/15 p-4 text-[#F3ECDA] ring-1 ring-inset ring-[#F3ECDA]/20">
                    Policy Compliance
                  </div>
                  <div className="rounded-lg bg-[#F3ECDA]/15 p-4 text-[#F3ECDA] ring-1 ring-inset ring-[#F3ECDA]/20">
                    AI Insights
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-[#c4ac95]/40 bg-[#F3ECDA]">
        <div className="mx-auto max-w-7xl px-4 py-8 text-sm text-[#94553D] sm:px-6 lg:px-8">
          © {new Date().getFullYear()} NYAYA SAMITI. All rights reserved.
        </div>
      </footer>
    </div>
  )
}

export default Homepage


