import {BrowserRouter as Router, Routes, Route} from "react-router-dom";
import './index.css'
import Homepage from './pages/Homepage'
import DocumentValidation from './pages/DocumentValidation'
import UserRegistration from './pages/UserRegistration'
import Dashboard from './pages/Dashboard'
import Profile from './pages/Profile'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/document-validation" element={<DocumentValidation />} />
        <Route path="/user-registration" element={<UserRegistration />} />
      </Routes>
    </Router>
  )
}

export default App
