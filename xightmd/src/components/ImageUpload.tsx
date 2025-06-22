'use client';

import { useState, useRef, DragEvent } from 'react';

interface ImageUploadProps {
  onUpload: (file: File, description?: string, priority?: string) => void;
  isAnalyzing: boolean;
}

export default function ImageUpload({ onUpload, isAnalyzing }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [userDescription, setUserDescription] = useState('');
  const [prioritySearch, setPrioritySearch] = useState('');
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onUpload(selectedFile, userDescription.trim() || undefined, prioritySearch.trim() || undefined);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setUserDescription('');
    setPrioritySearch('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const commonConditions = [
    'pneumonia', 'pneumothorax', 'pleural effusion', 'atelectasis', 
    'cardiomegaly', 'consolidation', 'mass', 'nodule', 'fracture'
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Chest X-ray</h2>
      
      {!selectedFile ? (
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
            dragActive 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={isAnalyzing}
          />
          
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            
            <div>
              <p className="text-lg font-medium text-gray-700">
                Drop your chest X-ray here, or{' '}
                <span className="text-blue-600 hover:text-blue-500 cursor-pointer">browse</span>
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Supports PNG, JPG, JPEG files up to 10MB
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Image Preview */}
          <div className="relative">
            <img
              src={previewUrl}
              alt="Chest X-ray preview"
              className="w-full max-h-96 object-contain rounded-lg border"
            />
            <button
              onClick={handleReset}
              className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 hover:bg-red-600 transition-colors"
              disabled={isAnalyzing}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* File Info */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">File Information</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Name:</span>
                <p className="font-medium truncate">{selectedFile.name}</p>
              </div>
              <div>
                <span className="text-gray-500">Size:</span>
                <p className="font-medium">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            </div>
          </div>

          {/* Advanced Options Toggle */}
          <div className="border-t pt-4">
            <button
              onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              className="flex items-center space-x-2 text-blue-600 hover:text-blue-700 font-medium"
            >
              <svg 
                className={`w-4 h-4 transition-transform ${showAdvancedOptions ? 'rotate-90' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              <span>Advanced Analysis Options</span>
            </button>
            
            {showAdvancedOptions && (
              <div className="mt-4 space-y-4 bg-blue-50 p-4 rounded-lg">
                {/* User Description */}
                <div>
                  <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
                    Patient Symptoms or Clinical History
                  </label>
                  <textarea
                    id="description"
                    rows={3}
                    value={userDescription}
                    onChange={(e) => setUserDescription(e.target.value)}
                    placeholder="Describe symptoms, concerns, or clinical history (optional)..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={isAnalyzing}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Example: "Patient complains of chest pain and shortness of breath"
                  </p>
                </div>

                {/* Priority Search */}
                <div>
                  <label htmlFor="priority" className="block text-sm font-medium text-gray-700 mb-2">
                    Specific Areas of Interest
                  </label>
                  <input
                    type="text"
                    id="priority"
                    value={prioritySearch}
                    onChange={(e) => setPrioritySearch(e.target.value)}
                    placeholder="What should we specifically look for? (optional)"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={isAnalyzing}
                  />
                  
                  {/* Quick Selection Buttons */}
                  <div className="mt-2">
                    <p className="text-xs text-gray-500 mb-2">Quick selections:</p>
                    <div className="flex flex-wrap gap-2">
                      {commonConditions.map((condition) => (
                        <button
                          key={condition}
                          type="button"
                          onClick={() => setPrioritySearch(condition)}
                          className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded-full transition-colors capitalize"
                          disabled={isAnalyzing}
                        >
                          {condition}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Info Box */}
                <div className="bg-blue-100 border border-blue-200 rounded-lg p-3">
                  <div className="flex items-start space-x-2">
                    <svg className="w-5 h-5 text-blue-600 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div className="text-sm text-blue-800">
                      <p className="font-medium">Enhanced AI Analysis</p>
                      <p>Providing additional context helps our AI generate more accurate and relevant reports.</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-4">
            <button
              onClick={handleSubmit}
              disabled={isAnalyzing}
              className={`flex-1 py-3 px-6 rounded-lg font-medium transition-all duration-300 ${
                isAnalyzing
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-1'
              }`}
            >
              {isAnalyzing ? (
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Analyzing...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <span>Analyze X-ray</span>
                </div>
              )}
            </button>
            
            <button
              onClick={handleReset}
              disabled={isAnalyzing}
              className="px-6 py-3 border border-gray-300 rounded-lg font-medium text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              Reset
            </button>
          </div>
        </div>
      )}

      {/* Processing Status */}
      {isAnalyzing && (
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            </div>
            <p className="text-blue-700 font-medium">AI agents are analyzing your chest X-ray...</p>
          </div>
          <div className="mt-3 space-y-1 text-blue-600 text-sm">
            <p>🔍 Triage Agent: Detecting abnormalities and assessing urgency</p>
            <p>📄 Report Agent: Generating structured radiology report with Claude AI</p>
            <p>✅ QA Agent: Validating findings and ensuring quality</p>
          </div>
          <p className="text-blue-600 text-sm mt-2">
            This typically takes 30-60 seconds for comprehensive analysis.
          </p>
        </div>
      )}
    </div>
  );
  }