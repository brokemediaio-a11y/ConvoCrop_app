'use client'

import { useState, useRef } from 'react'
import GlassSurface from './GlassSurface'

interface ImageUploadProps {
  onImageSelect: (imageBase64: string) => void
}

export default function ImageUpload({ onImageSelect }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const MAX_INPUT_SIZE_BYTES = 10 * 1024 * 1024
  const TARGET_UPLOAD_SIZE_BYTES = 6 * 1024 * 1024
  const SUPPORTED_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp'])

  async function fileToDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onerror = () => reject(new Error('Failed to read image file.'))
      reader.onloadend = () => resolve(String(reader.result ?? ''))
      reader.readAsDataURL(file)
    })
  }

  async function dataUrlToImage(dataUrl: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error('Could not decode this image.'))
      img.src = dataUrl
    })
  }

  async function compressToJpeg(file: File, maxBytes: number): Promise<string> {
    const dataUrl = await fileToDataUrl(file)
    const img = await dataUrlToImage(dataUrl)
    const canvas = document.createElement('canvas')

    const maxSide = 2048
    const scale = Math.min(1, maxSide / Math.max(img.width, img.height))
    canvas.width = Math.max(1, Math.round(img.width * scale))
    canvas.height = Math.max(1, Math.round(img.height * scale))
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Image compression unavailable in this browser.')

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    let quality = 0.88
    let output = canvas.toDataURL('image/jpeg', quality)
    while (output.length * 0.75 > maxBytes && quality > 0.42) {
      quality -= 0.08
      output = canvas.toDataURL('image/jpeg', quality)
    }
    return output
  }

  const handleFile = async (file: File) => {
    setError(null)

    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file.')
      return
    }

    if (!SUPPORTED_TYPES.has(file.type)) {
      setError('Unsupported format. Please use JPG, PNG, or WebP. HEIC, TIFF, and BMP are not supported yet.')
      return
    }

    if (file.size > MAX_INPUT_SIZE_BYTES) {
      setError('Image is too large (over 10MB). Please upload a smaller photo.')
      return
    }

    setIsProcessing(true)
    try {
      let base64String = await fileToDataUrl(file)
      if (file.size > TARGET_UPLOAD_SIZE_BYTES) {
        base64String = await compressToJpeg(file, TARGET_UPLOAD_SIZE_BYTES)
      }
      const base64 = base64String.includes(',') ? base64String.split(',')[1] : base64String
      setPreview(base64String)
      onImageSelect(base64)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to process image.'
      setError(msg)
    } finally {
      setIsProcessing(false)
    }
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
      void handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      void handleFile(e.target.files[0])
    }
  }

  const onButtonClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="w-full">
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          dragActive
            ? 'border-[#00a71b] bg-[#00a71b]/5'
            : 'border-[#252525]/20 hover:border-[#00a71b]/40'
        }`}
        style={{ background: dragActive ? undefined : 'rgba(255,255,255,0.3)', backdropFilter: 'blur(8px)' }}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png,image/webp"
          onChange={handleChange}
          className="hidden"
        />

        {preview ? (
          <div className="space-y-4">
            <img
              src={preview}
              alt="Preview"
              className="max-w-full max-h-64 mx-auto rounded-lg shadow-md border-2 border-[#00a71b]/20"
            />
            <GlassSurface
              width={160}
              height={42}
              borderRadius={12}
              onClick={onButtonClick}
              className={`mx-auto ${isProcessing ? 'pointer-events-none opacity-70' : 'cursor-pointer'}`}
            >
              <span className="text-[#252525] font-medium text-sm">{isProcessing ? 'Processing...' : 'Change Image'}</span>
            </GlassSurface>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex justify-center">
              <svg
                className="w-14 h-14 text-[#00a71b]/60"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <p className="text-base font-medium text-[#252525] mb-1">
                Drop your image here
              </p>
              <p className="text-xs text-[#252525]/50 mb-4">
                or click to browse
              </p>
              <GlassSurface
                width={160}
                height={44}
                borderRadius={12}
                onClick={onButtonClick}
                className={`mx-auto ${isProcessing ? 'pointer-events-none opacity-70' : 'cursor-pointer'}`}
              >
                <span className="text-[#252525] font-medium text-sm">{isProcessing ? 'Processing...' : 'Choose Image'}</span>
              </GlassSurface>
            </div>
            <p className="text-[10px] text-[#252525]/35">
              JPG, PNG, WebP -- up to 10MB input (auto-compressed when needed)
            </p>
          </div>
        )}
      </div>
      {error && <p className="mt-2 text-xs text-red-700">{error}</p>}
    </div>
  )
}
