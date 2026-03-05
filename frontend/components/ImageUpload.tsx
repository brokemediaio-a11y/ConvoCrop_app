'use client'

import { useState, useRef } from 'react'
import GlassSurface from './GlassSurface'

interface ImageUploadProps {
  onImageSelect: (imageBase64: string) => void
}

export default function ImageUpload({ onImageSelect }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file')
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      alert('Image size should be less than 10MB')
      return
    }

    const reader = new FileReader()
    reader.onloadend = () => {
      const base64String = reader.result as string
      const base64 = base64String.includes(',')
        ? base64String.split(',')[1]
        : base64String
      setPreview(base64String)
      onImageSelect(base64)
    }
    reader.readAsDataURL(file)
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const onButtonClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="w-full">
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-[#00a71b] bg-[#00a71b]/10'
            : 'border-[#00a71b]/40 bg-[#212121] hover:border-[#00a71b]/60'
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
          className="hidden"
        />

        {preview ? (
          <div className="space-y-4">
            <img
              src={preview}
              alt="Preview"
              className="max-w-full max-h-64 mx-auto rounded-lg shadow-md border-2 border-[#00a71b]/30"
            />
            <div className="flex gap-2 justify-center">
              <GlassSurface
                width={140}
                height={40}
                borderRadius={20}
                onClick={onButtonClick}
                className="cursor-pointer"
              >
                <span className="text-[#dedede] font-medium">Change Image</span>
              </GlassSurface>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex justify-center">
              <svg
                className="w-16 h-16 text-[#00a71b]"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <p className="text-lg font-medium text-[#dedede] mb-2">
                Upload Rice Leaf Image
              </p>
              <p className="text-sm text-[#dedede] opacity-70 mb-4">
                Drag and drop an image here, or click to select
              </p>
              <GlassSurface
                width={160}
                height={45}
                borderRadius={22}
                onClick={onButtonClick}
                className="cursor-pointer mx-auto"
              >
                <span className="text-[#dedede] font-medium">Choose Image</span>
              </GlassSurface>
            </div>
            <p className="text-xs text-[#dedede] opacity-50">
              Supported formats: JPEG, PNG, WebP (Max 10MB)
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
