import "../index.css";
import { useState } from "react";
import { Link } from "react-router-dom";
import apiService from "../services/api";
import Footer from "../components/Footer";
import FloatingChatbot from "../components/FloatingChatbot";

function UserRegistration() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    mobno: "",
    age: "",
    address: "",
    no_of_family_members: 1,
    family_member: [
      {
        name: "",
        email: "",
        mobno: "",
        age: "",
        relation: "",
        address: "",
        video_urls: ""
      }
    ]
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);
  const [errors, setErrors] = useState({});

  const handleInputChange = (e, index = null) => {
    const { name, value } = e.target;
    
    if (index !== null) {
      // Handle family member fields
      setFormData(prev => ({
        ...prev,
        family_member: prev.family_member.map((member, i) => 
          i === index ? { ...member, [name]: value } : member
        )
      }));
    } else {
      // Handle main user fields
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }

    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ""
      }));
    }
  };

  const addFamilyMember = () => {
    setFormData(prev => ({
      ...prev,
      family_member: [
        ...prev.family_member,
        {
          name: "",
          email: "",
          mobno: "",
          age: "",
          relation: "",
          address: "",
          video_urls: ""
        }
      ],
      no_of_family_members: prev.family_member.length + 1
    }));
  };

  const removeFamilyMember = (index) => {
    if (formData.family_member.length > 1) {
      setFormData(prev => ({
        ...prev,
        family_member: prev.family_member.filter((_, i) => i !== index),
        no_of_family_members: prev.family_member.length - 1
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    // Validate main user fields
    if (!formData.name.trim()) newErrors.name = "Name is required";
    if (!formData.email.trim()) newErrors.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = "Email is invalid";
    if (!formData.mobno.trim()) newErrors.mobno = "Mobile number is required";
    else if (formData.mobno.length < 10) newErrors.mobno = "Mobile number must be at least 10 digits";
    if (!formData.age) newErrors.age = "Age is required";
    else if (parseInt(formData.age) <= 0) newErrors.age = "Age must be greater than 0";
    if (!formData.address.trim()) newErrors.address = "Address is required";
    else if (formData.address.length < 10) newErrors.address = "Address must be at least 10 characters";

    // Validate family members
    formData.family_member.forEach((member, index) => {
      if (!member.name.trim()) newErrors[`family_${index}_name`] = "Family member name is required";
      if (!member.email.trim()) newErrors[`family_${index}_email`] = "Family member email is required";
      else if (!/\S+@\S+\.\S+/.test(member.email)) newErrors[`family_${index}_email`] = "Family member email is invalid";
      if (!member.mobno.trim()) newErrors[`family_${index}_mobno`] = "Family member mobile number is required";
      else if (member.mobno.length < 10) newErrors[`family_${index}_mobno`] = "Family member mobile number must be at least 10 digits";
      if (!member.age) newErrors[`family_${index}_age`] = "Family member age is required";
      else if (parseInt(member.age) <= 0) newErrors[`family_${index}_age`] = "Family member age must be greater than 0";
      if (!member.relation.trim()) newErrors[`family_${index}_relation`] = "Family member relation is required";
      if (!member.address.trim()) newErrors[`family_${index}_address`] = "Family member address is required";
      else if (member.address.length < 10) newErrors[`family_${index}_address`] = "Family member address must be at least 10 characters";
      if (!member.video_urls.trim()) newErrors[`family_${index}_video_urls`] = "Family member video URL is required";
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      setSubmitStatus("error");
      return;
    }

    setIsSubmitting(true);
    setSubmitStatus(null);

    try {
      const result = await apiService.createUser(formData);
      setSubmitStatus("success");
      console.log("User created successfully:", result);
      
      // Reset form
      setFormData({
        name: "",
        email: "",
        mobno: "",
        age: "",
        address: "",
        no_of_family_members: 1,
        family_member: [
          {
            name: "",
            email: "",
            mobno: "",
            age: "",
            relation: "",
            address: "",
            video_urls: ""
          }
        ]
      });
    } catch (error) {
      setSubmitStatus("error");
      console.error("Error creating user:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F3ECDA]/80 text-[#2b1d14]">
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
              <Link
                to="/document-validation"
                className="text-sm font-medium text-[#F3ECDA] hover: hover:scale-105 transition-all duration-300 px-4 py-2 rounded-md"
              >
                Document Validation
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="bg-[#F3ECDA]/70 py-12">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-[#94553D] mb-4">
              User Registration
            </h1>
            <p className="text-lg text-[#94553D]/80">
              Register yourself and your family members for legal assistance
            </p>
          </div>

          <div className="bg-white/90 rounded-lg shadow-lg p-8">
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Main User Information */}
              <div className="border-b border-gray-200 pb-8">
                <h2 className="text-2xl font-semibold text-[#94553D] mb-6">
                  Personal Information
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                      Full Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                        errors.name ? "border-red-500" : "border-gray-300"
                      }`}
                      placeholder="Enter your full name"
                    />
                    {errors.name && <p className="mt-1 text-sm text-red-600">{errors.name}</p>}
                  </div>

                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                      Email Address *
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                        errors.email ? "border-red-500" : "border-gray-300"
                      }`}
                      placeholder="Enter your email address"
                    />
                    {errors.email && <p className="mt-1 text-sm text-red-600">{errors.email}</p>}
                  </div>

                  <div>
                    <label htmlFor="mobno" className="block text-sm font-medium text-gray-700 mb-2">
                      Mobile Number *
                    </label>
                    <input
                      type="tel"
                      id="mobno"
                      name="mobno"
                      value={formData.mobno}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                        errors.mobno ? "border-red-500" : "border-gray-300"
                      }`}
                      placeholder="Enter your mobile number"
                    />
                    {errors.mobno && <p className="mt-1 text-sm text-red-600">{errors.mobno}</p>}
                  </div>

                  <div>
                    <label htmlFor="age" className="block text-sm font-medium text-gray-700 mb-2">
                      Age *
                    </label>
                    <input
                      type="number"
                      id="age"
                      name="age"
                      value={formData.age}
                      onChange={handleInputChange}
                      min="1"
                      className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                        errors.age ? "border-red-500" : "border-gray-300"
                      }`}
                      placeholder="Enter your age"
                    />
                    {errors.age && <p className="mt-1 text-sm text-red-600">{errors.age}</p>}
                  </div>

                  <div className="md:col-span-2">
                    <label htmlFor="address" className="block text-sm font-medium text-gray-700 mb-2">
                      Address *
                    </label>
                    <textarea
                      id="address"
                      name="address"
                      value={formData.address}
                      onChange={handleInputChange}
                      rows="3"
                      className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                        errors.address ? "border-red-500" : "border-gray-300"
                      }`}
                      placeholder="Enter your complete address"
                    />
                    {errors.address && <p className="mt-1 text-sm text-red-600">{errors.address}</p>}
                  </div>
                </div>
              </div>

              {/* Family Members */}
              <div>
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-semibold text-[#94553D]">
                    Family Members
                  </h2>
                  <button
                    type="button"
                    onClick={addFamilyMember}
                    className="bg-[#94553D] text-white px-4 py-2 rounded-lg hover:bg-[#7a3d2f] transition-colors"
                  >
                    Add Family Member
                  </button>
                </div>

                <div className="space-y-8">
                  {formData.family_member.map((member, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-6 relative">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-medium text-[#94553D]">
                          Family Member {index + 1}
                        </h3>
                        {formData.family_member.length > 1 && (
                          <button
                            type="button"
                            onClick={() => removeFamilyMember(index)}
                            className="text-red-600 hover:text-red-800 text-sm font-medium"
                          >
                            Remove
                          </button>
                        )}
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Name *
                          </label>
                          <input
                            type="text"
                            name="name"
                            value={member.name}
                            onChange={(e) => handleInputChange(e, index)}
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                              errors[`family_${index}_name`] ? "border-red-500" : "border-gray-300"
                            }`}
                            placeholder="Enter family member name"
                          />
                          {errors[`family_${index}_name`] && (
                            <p className="mt-1 text-sm text-red-600">{errors[`family_${index}_name`]}</p>
                          )}
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Email *
                          </label>
                          <input
                            type="email"
                            name="email"
                            value={member.email}
                            onChange={(e) => handleInputChange(e, index)}
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                              errors[`family_${index}_email`] ? "border-red-500" : "border-gray-300"
                            }`}
                            placeholder="Enter family member email"
                          />
                          {errors[`family_${index}_email`] && (
                            <p className="mt-1 text-sm text-red-600">{errors[`family_${index}_email`]}</p>
                          )}
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Mobile Number *
                          </label>
                          <input
                            type="tel"
                            name="mobno"
                            value={member.mobno}
                            onChange={(e) => handleInputChange(e, index)}
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                              errors[`family_${index}_mobno`] ? "border-red-500" : "border-gray-300"
                            }`}
                            placeholder="Enter family member mobile number"
                          />
                          {errors[`family_${index}_mobno`] && (
                            <p className="mt-1 text-sm text-red-600">{errors[`family_${index}_mobno`]}</p>
                          )}
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Age *
                          </label>
                          <input
                            type="number"
                            name="age"
                            value={member.age}
                            onChange={(e) => handleInputChange(e, index)}
                            min="1"
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                              errors[`family_${index}_age`] ? "border-red-500" : "border-gray-300"
                            }`}
                            placeholder="Enter family member age"
                          />
                          {errors[`family_${index}_age`] && (
                            <p className="mt-1 text-sm text-red-600">{errors[`family_${index}_age`]}</p>
                          )}
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Relation *
                          </label>
                          <select
                            name="relation"
                            value={member.relation}
                            onChange={(e) => handleInputChange(e, index)}
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                              errors[`family_${index}_relation`] ? "border-red-500" : "border-gray-300"
                            }`}
                          >
                            <option value="">Select relation</option>
                            <option value="Spouse">Spouse</option>
                            <option value="Child">Child</option>
                            <option value="Parent">Parent</option>
                            <option value="Sibling">Sibling</option>
                            <option value="Other">Other</option>
                          </select>
                          {errors[`family_${index}_relation`] && (
                            <p className="mt-1 text-sm text-red-600">{errors[`family_${index}_relation`]}</p>
                          )}
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Video URL *
                          </label>
                          <input
                            type="url"
                            name="video_urls"
                            value={member.video_urls}
                            onChange={(e) => handleInputChange(e, index)}
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                              errors[`family_${index}_video_urls`] ? "border-red-500" : "border-gray-300"
                            }`}
                            placeholder="Enter video URL"
                          />
                          {errors[`family_${index}_video_urls`] && (
                            <p className="mt-1 text-sm text-red-600">{errors[`family_${index}_video_urls`]}</p>
                          )}
                        </div>

                        <div className="md:col-span-2">
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Address *
                          </label>
                          <textarea
                            name="address"
                            value={member.address}
                            onChange={(e) => handleInputChange(e, index)}
                            rows="2"
                            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-[#94553D] focus:border-transparent ${
                              errors[`family_${index}_address`] ? "border-red-500" : "border-gray-300"
                            }`}
                            placeholder="Enter family member address"
                          />
                          {errors[`family_${index}_address`] && (
                            <p className="mt-1 text-sm text-red-600">{errors[`family_${index}_address`]}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Submit Button */}
              <div className="flex items-center justify-center pt-6">
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="bg-[#94553D] text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-[#7a3d2f] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSubmitting ? "Registering..." : "Register User"}
                </button>
              </div>

              {/* Status Messages */}
              {submitStatus === "success" && (
                <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded-lg">
                  <div className="flex items-center">
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    User registered successfully!
                  </div>
                </div>
              )}

              {submitStatus === "error" && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">
                  <div className="flex items-center">
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    Error registering user. Please check your information and try again.
                  </div>
                </div>
              )}
            </form>
          </div>
        </div>
      </main>

      <Footer />
      <FloatingChatbot />
    </div>
  );
}

export default UserRegistration;
