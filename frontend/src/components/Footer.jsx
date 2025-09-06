export default function Footer() {
  return (
    <footer className="border-t border-[#c4ac95]/40 bg-[#94553D]">
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-[#F3ECDA]">
          {/* About NYAYA SAMITI - Left */}
          <div className="space-y-3">
            <div className="text-xl font-semibold tracking-wide">NYAYA SAMITI</div>
            <p className="text-sm opacity-90 leading-relaxed">
              AI-powered legal document verification and guidance platform for Indian law and governance.
            </p>
            <p className="text-xs opacity-80 leading-relaxed">
              Helping citizens and institutions validate documents, understand procedures, and navigate compliance.
            </p>
          </div>

          {/* Contributors - Middle */}
          <div className="space-y-3 text-center">
            <div className="text-lg font-semibold">Contributors</div>
            <div className="text-sm space-y-1">
              <div>Shravya H Jain</div>
              <div>Sakshi Shetty</div>
              <div>Srishyla Kumar TP</div>
              <div>Shreesha Shetty</div>
            </div>
          </div>

          {/* Social Media - Right */}
          <div className="space-y-3 text-center md:text-right">
            <div className="text-lg font-semibold">Follow Us</div>
            <div className="flex items-center justify-center md:justify-end gap-4">
              <a href="#" aria-label="Twitter" className="text-[#F3ECDA] hover:opacity-80 transition-opacity">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-6 w-6">
                  <path d="M19.633 7.997c.013.176.013.353.013.53 0 5.397-4.11 11.621-11.62 11.621-2.308 0-4.456-.675-6.262-1.84.323.038.632.051.968.051a8.21 8.21 0 0 0 5.09-1.753 4.105 4.105 0 0 1-3.833-2.845c.255.038.51.064.778.064.374 0 .747-.051 1.095-.14A4.098 4.098 0 0 1 2.8 9.902v-.051c.546.304 1.182.49 1.855.515A4.093 4.093 0 0 1 2.8 7.007c0-.765.204-1.47.56-2.082a11.65 11.65 0 0 0 8.457 4.29 4.62 4.62 0 0 1-.102-.94 4.1 4.1 0 0 1 7.1-2.803 8.09 8.09 0 0 0 2.603-.992 4.11 4.11 0 0 1-1.803 2.263A8.21 8.21 0 0 0 22 6.835a8.818 8.818 0 0 1-2.367 2.162z" />
                </svg>
              </a>
              <a href="#" aria-label="GitHub" className="text-[#F3ECDA] hover:opacity-80 transition-opacity">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-6 w-6">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.426 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.009-.866-.014-1.7-2.782.605-3.37-1.342-3.37-1.342-.454-1.155-1.11-1.464-1.11-1.464-.907-.62.069-.607.069-.607 1.003.07 1.532 1.032 1.532 1.032.892 1.53 2.341 1.088 2.91.833.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.338 1.909-1.296 2.748-1.026 2.748-1.026.546 1.378.203 2.397.1 2.65.64.7 1.028 1.595 1.028 2.688 0 3.847-2.338 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.416-.012 2.744 0 .268.18.58.688.481A10.02 10.02 0 0 0 22 12.017C22 6.484 17.523 2 12 2z" clipRule="evenodd" />
                </svg>
              </a>
              <a href="#" aria-label="LinkedIn" className="text-[#F3ECDA] hover:opacity-80 transition-opacity">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-6 w-6">
                  <path d="M19 3A2.94 2.94 0 0 1 22 6v12a2.94 2.94 0 0 1-3 3H5a2.94 2.94 0 0 1-3-3V6a2.94 2.94 0 0 1 3-3Zm-9.5 6H7V19h2.5Zm-1.25-1.75A1.25 1.25 0 1 0 8.5 6a1.25 1.25 0 0 0-.25 2.5ZM19 13.25c0-2.5-1.34-3.75-3.12-3.75a2.66 2.66 0 0 0-2.38 1.3h-.06V9H11V19h2.5v-5.25c0-1.38.5-2.25 1.75-2.25 1.19 0 1.75.84 1.75 2.25V19H19Z" />
                </svg>
              </a>
            </div>
          </div>
        </div>

        {/* Copyright - Bottom */}
        <div className="mt-8 pt-6 border-t border-[#c4ac95]/20 text-center">
          <div className="text-xs opacity-80">
            © {new Date().getFullYear()} NYAYA SAMITI. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  )
}


