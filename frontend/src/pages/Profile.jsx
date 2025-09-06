import "../index.css";
import { Link } from "react-router-dom";
import Footer from "../components/Footer";
import FloatingChatbot from "../components/FloatingChatbot";

export default function Profile() {
  return (
    <div className="min-h-screen bg-[#F3ECDA]/80 text-[#94553D]">
      <header className="border-b border-[#c4ac95]/40 bg-[#94553D]">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-full bg-[#F3ECDA]"></div>
              <span className="text-xl font-semibold tracking-wide text-[#F3ECDA]">
                NYAYA SAMITI
              </span>
            </div>
            <nav className="flex items-center gap-4">
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

      <main className="bg-[#F3ECDA]/70">
        <section className="relative isolate">
          <div className="absolute inset-0 bg-[#F3ECDA]/60 text-[#94553D]"></div>
          <div className="relative mx-auto grid max-w-7xl  gap-12 px-4 py-24 sm:px-6 lg:lg:gap-16 lg:px-8">
          <div className="text-center items-center justify-center">
            <h1 className="text-4xl font-bold text-[#94553D] mb-4">Profile</h1>
            <p className="text-lg text-[#94553D]/80 ">
              Manage your profile 
            </p>
          </div>
          </div>
          
        </section>
      </main>

      <Footer />
      <FloatingChatbot />
    </div>
  );
}
