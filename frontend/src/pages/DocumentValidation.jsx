import "../index.css";
import { useState, useRef } from "react";
import { Link } from "react-router-dom";

function DocumentValidation() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  const fileInputRef = useRef(null);

  function handleFileChange(event) {
    const file = event.target.files && event.target.files[0];
    setResult(null);
    setErrorMessage("");
    if (file) {
      setSelectedFile(file);
    }
  }


  async function handleAnalyze() {
    if (!selectedFile) {
      setErrorMessage("Please upload a document to validate.");
      return;
    }
    setIsAnalyzing(true);
    setErrorMessage("");
    
    try {
      // Placeholder for backend/ML integration
      await new Promise((r) => setTimeout(r, 1200));
      setResult({
        isAuthentic: true,
        confidence: 0.93,
        extractedEntities: [
          { label: "Name", value: "John Doe" },
          { label: "Document ID", value: "ABC-123-XYZ" },
          { label: "Issue Date", value: "2024-11-12" },
        ],
        notes: "No forgery patterns detected. OCR text consistent with layout.",
      });
    } catch {
      setErrorMessage("Failed to analyze document. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#F3ECDA] text-[#2b1d14]">
      <header className="border-b border-[#c4ac95]/40 bg-[#94553D]">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
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
              <Link
                to="/dashboard"
                className="text-sm font-medium text-[#F3ECDA] hover: hover:scale-105 transition-all duration-300 px-4 py-2 rounded-md"
              >
                Dashboard
              </Link>
             
            </nav>
          </div>
        </div>
      </header>

      <main className="bg-[#F3ECDA]">
        <section className="relative isolate">
        <div className="absolute inset-0 bg-[#94553D]"></div>
        <div className="relative w-full px-4 py-16 sm:px-6 lg:px-10">
            <h1 className="text-5xl font-extrabold tracking-tight text-[#F3ECDA] sm:text-6xl">
              Document Validation
            </h1>
            <p className="mt-4 max-w-5xl text-lg leading-8 text-[#F3ECDA]/95">
              Upload a legal document to verify authenticity, extract key details, and assess risk using AI.
            </p>

            <div className="mt-6 grid gap-4 sm:grid">
              <div className="sm:col-span-2">
                {!selectedFile && (
                  <div
                    onClick={() => fileInputRef.current && fileInputRef.current.click()}
                    className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-[#F3ECDA]/40 bg-[#F3ECDA]/10 px-6 py-10 text-center text-[#F3ECDA] transition hover:bg-[#F3ECDA]/15"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                      className="h-12 w-12 opacity-90"
                    >
                      <path d="M3 16.5A3.5 3.5 0 0 0 6.5 20H17a4 4 0 0 0 1-7.874V12a5 5 0 0 0-9.584-2.134A4 4 0 0 0 3 13v3.5Z"/>
                    </svg>
                    <div className="mt-3 pb-2 text-base font-semibold">Click to choose a file</div>
                    <div className="text-xs opacity-80">or drag & drop (PDF, PNG, JPG)</div>
                  </div>
                )}
                <input
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg"
                  className="hidden"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                {selectedFile && (
                  <div className="mt-4 rounded-lg border border-[#F3ECDA]/30 bg-[#F3ECDA]/10 p-4">
                    <div className="flex items-center gap-3">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                        className="h-8 w-8 text-[#F3ECDA]"
                      >
                        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
                      </svg>
                      <div className="flex-1">
                        <div className="text-sm font-medium text-[#F3ECDA]">
                          {selectedFile.name}
                        </div>
                        <div className="text-xs text-[#F3ECDA]/70">
                          {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • {selectedFile.type}
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          setSelectedFile(null);
                          setResult(null);
                          setErrorMessage("");
                          if (fileInputRef.current) {
                            fileInputRef.current.value = "";
                          }
                        }}
                        className="rounded-md p-1 text-[#F3ECDA]/70 hover:bg-[#F3ECDA]/20 hover:text-[#F3ECDA]"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 24 24"
                          fill="currentColor"
                          className="h-4 w-4"
                        >
                          <path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
                        </svg>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
            {selectedFile && (
              <div className="flex items-end">
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full mt-2 rounded-md bg-[#FFCBA4] px-5 py-2.5 text-sm font-semibold text-[#94553D] shadow-sm transition hover:bg-[#f7b983] disabled:opacity-60"
                >
                  {isAnalyzing ? "Analyzing…" : "Validate Document"}
                </button>
              </div>
            )}

            {errorMessage && (
              <div className="mt-4 rounded-md border border-red-300/40 bg-red-100/20 p-3 text-sm text-red-900/90">
                {errorMessage}
              </div>
            )}

            {result && (
              <div className="mt-10">
                <div className="bg-white/10 backdrop-blur-sm border border-[#F3ECDA]/20 rounded-xl p-8 shadow-lg">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <h2 className="text-2xl font-bold text-[#F3ECDA]">Validation Results</h2>
                  </div>
                  
                  <div className="space-y-6 text-[#F3ECDA]">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-[#F3ECDA]/10 rounded-lg p-4 border border-[#F3ECDA]/20">
                        <div className="text-sm uppercase tracking-wide opacity-80 mb-2">Authenticity</div>
                        <div className="text-2xl font-bold flex items-center gap-2">
                          {result.isAuthentic ? (
                            <>
                              <span className="text-green-400">✓</span>
                              Authentic
                            </>
                          ) : (
                            <>
                              <span className="text-red-400">⚠</span>
                              Suspicious
                            </>
                          )}
                        </div>
                      </div>
                      
                      <div className="bg-[#F3ECDA]/10 rounded-lg p-4 border border-[#F3ECDA]/20">
                        <div className="text-sm uppercase tracking-wide opacity-80 mb-2">Confidence</div>
                        <div className="text-2xl font-bold flex items-center gap-2">
                          <span className="text-blue-400">📊</span>
                          {Math.round(result.confidence * 100)}%
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-[#F3ECDA]/10 rounded-lg p-4 border border-[#F3ECDA]/20">
                      <div className="text-sm uppercase tracking-wide opacity-80 mb-2">Analysis Notes</div>
                      <p className="text-lg leading-7">{result.notes}</p>
                    </div>
                    
                    <div className="bg-[#F3ECDA]/10 rounded-lg p-4 border border-[#F3ECDA]/20">
                      <div className="text-sm uppercase tracking-wide opacity-80 mb-4">Extracted Entities</div>
                      <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2 md:grid-cols-3">
                        {result.extractedEntities.map((item, index) => (
                          <div key={index} className="flex flex-col">
                            <dt className="text-xs opacity-80 mb-1">{item.label}</dt>
                            <dd className="text-base font-medium bg-[#F3ECDA]/20 rounded px-2 py-1">{item.value}</dd>
                          </div>
                        ))}
                      </dl>
                    </div>
                  </div>
                  
                  {/* User Registration Button - Only show if document is valid */}
                  {result.isAuthentic && (
                    <div className="mt-8 flex justify-center">
                      <Link
                        to="/user-registration"
                        className="rounded-md bg-[#FFCBA4] px-8 py-3 text-lg font-semibold text-[#94553D] shadow-sm transition hover:bg-[#f7b983] hover:shadow-md"
                      >
                        Proceed to User Registration
                      </Link>
                    </div>
                  )}
                </div>
              </div>
            )}
            {/* <div className="mt-12 flex items-center align-center justify-center gap-4">
              <Link
                to="/validate-and-file"
                className="rounded-md bg-[#FFCBA4] px-6 py-3 text-base font-semibold text-[#94553D] shadow-sm transition hover:bg-[#f7b983]"
              >
                Validate & File a Case
              </Link>
              <Link
                to="/"
                className="rounded-md px-6 py-3 text-base font-semibold text-[#F3ECDA] ring-1 ring-inset ring-[#F3ECDA]/40 transition hover:bg-[#F3ECDA]/10"
              >
                Back to Home
              </Link>
            </div> */}

            <div className="mt-14">
              <h2 className="text-3xl font-bold text-[#F3ECDA]">How our verification works</h2>
              <p className="mt-3 max-w-6xl text-lg leading-8 text-[#F3ECDA]/95">
                We combine OCR, forgery detection, and policy checks to validate legal documents end-to-end.
              </p>
              <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                <figure>
                  <img src="https://images.unsplash.com/photo-1551836022-d5d88e9218df?q=80&w=1200&auto=format&fit=crop" alt="OCR pipeline" className="h-56 w-full rounded-md object-cover" />
                  <figcaption className="mt-2 text-sm text-[#F3ECDA]/90">OCR & entity extraction</figcaption>
                </figure>
                <figure>
                  <img src="https://i0.wp.com/alowaislaw.com/wp-content/uploads/2022/04/Forgery-Law-Alowais-Dubai-Law-Firm.jpg" alt="Forgery detection" className="h-56 w-full rounded-md object-cover" />
                  <figcaption className="mt-2 text-sm text-[#F3ECDA]/90">Forgery pattern detection</figcaption>
                </figure>
                <figure>
                  <img src="https://images.unsplash.com/photo-1526378722484-bd91ca387e72?q=80&w=1200&auto=format&fit=crop" alt="Policy compliance" className="h-56 w-full rounded-md object-cover" />
                  <figcaption className="mt-2 text-sm text-[#F3ECDA]/90">Policy & format checks</figcaption>
                </figure>
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
  );
}

export default DocumentValidation;
