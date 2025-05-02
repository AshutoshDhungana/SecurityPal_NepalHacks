"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Upload, File, CheckCircle, AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface FileUploaderProps {
  acceptedFileTypes: string
  maxFileSizeMB: number
  endpoint: string
  multiple?: boolean
}

export function FileUploader({ acceptedFileTypes, maxFileSizeMB, endpoint, multiple = false }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle")
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files)
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files)
    }
  }

  const handleFiles = (fileList: FileList) => {
    const newFiles: File[] = []
    let hasErrors = false

    // Reset previous errors
    setErrorMessage(null)

    // Check file types and sizes
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i]

      // Check file type
      const fileType = file.name.split(".").pop()?.toLowerCase()
      const isAcceptedType = acceptedFileTypes.includes(`.${fileType}`)

      if (!isAcceptedType) {
        setErrorMessage(`File type not accepted: ${file.name}. Please upload ${acceptedFileTypes} files.`)
        hasErrors = true
        continue
      }

      // Check file size
      if (file.size > maxFileSizeMB * 1024 * 1024) {
        setErrorMessage(`File too large: ${file.name}. Maximum size is ${maxFileSizeMB}MB.`)
        hasErrors = true
        continue
      }

      newFiles.push(file)
    }

    if (!hasErrors) {
      if (multiple) {
        setFiles((prev) => [...prev, ...newFiles])
      } else {
        setFiles(newFiles.slice(0, 1)) // Only take the first file if multiple is false
      }
    }
  }

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index))
  }

  const uploadFiles = async () => {
    if (files.length === 0) return

    setUploadStatus("uploading")
    setUploadProgress(0)

    const formData = new FormData()
    files.forEach((file) => {
      formData.append("files", file)
    })

    try {
      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev === null) return 0
          if (prev >= 95) {
            clearInterval(interval)
            return 95
          }
          return prev + 5
        })
      }, 100)

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // In a real implementation, you would use fetch:
      /*
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error("Upload failed")
      }
      */

      clearInterval(interval)
      setUploadProgress(100)
      setUploadStatus("success")

      // Clear files after successful upload
      setTimeout(() => {
        setFiles([])
        setUploadStatus("idle")
        setUploadProgress(null)
      }, 2000)
    } catch (error) {
      console.error("Upload error:", error)
      setUploadStatus("error")
      setErrorMessage("Failed to upload files. Please try again.")
    }
  }

  return (
    <div className="space-y-4">
      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
          isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25",
          uploadStatus === "error" ? "border-red-500 bg-red-50" : "",
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          accept={acceptedFileTypes}
          multiple={multiple}
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center space-y-2">
          <div className="rounded-full bg-primary/10 p-3">
            <Upload className="h-6 w-6 text-primary" />
          </div>
          <div className="text-lg font-medium">
            {uploadStatus === "error" ? "Upload Failed" : "Drop files here or click to upload"}
          </div>
          <div className="text-sm text-muted-foreground">
            {multiple ? "Upload multiple files" : "Upload a single file"} ({acceptedFileTypes}) up to {maxFileSizeMB}MB
          </div>

          {errorMessage && (
            <div className="flex items-center text-sm text-red-500 mt-2">
              <AlertCircle className="h-4 w-4 mr-1" />
              {errorMessage}
            </div>
          )}
        </div>
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          <div className="text-sm font-medium">Selected Files:</div>
          <div className="space-y-2">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between rounded-md border p-2">
                <div className="flex items-center space-x-2">
                  <File className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">{file.name}</span>
                  <span className="text-xs text-muted-foreground">({(file.size / (1024 * 1024)).toFixed(2)} MB)</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile(index)
                  }}
                >
                  Remove
                </Button>
              </div>
            ))}
          </div>

          {uploadProgress !== null && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="h-2" />
            </div>
          )}

          <div className="flex justify-end space-x-2">
            <Button
              variant="outline"
              onClick={(e) => {
                e.stopPropagation()
                setFiles([])
              }}
              disabled={uploadStatus === "uploading"}
            >
              Clear All
            </Button>
            <Button
              onClick={(e) => {
                e.stopPropagation()
                uploadFiles()
              }}
              disabled={uploadStatus === "uploading" || uploadStatus === "success"}
            >
              {uploadStatus === "uploading" ? (
                "Uploading..."
              ) : uploadStatus === "success" ? (
                <span className="flex items-center">
                  <CheckCircle className="h-4 w-4 mr-1" /> Uploaded
                </span>
              ) : (
                "Upload"
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
