"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Upload, File, CheckCircle, AlertCircle, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { apiService } from "@/lib/api-service"
import { useToast } from "@/hooks/use-toast"

export function CSVUploader() {
  const [isDragging, setIsDragging] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle")
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const { toast } = useToast()

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

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files))
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files))
    }
  }

  const handleFiles = (newFiles: File[]) => {
    // Filter for CSV files only
    const csvFiles = newFiles.filter((file) => file.name.endsWith(".csv"))

    if (csvFiles.length === 0) {
      setErrorMessage("Please upload CSV files only")
      return
    }

    if (newFiles.length !== csvFiles.length) {
      toast({
        title: "File type warning",
        description: "Some files were skipped because they were not CSV files",
        variant: "destructive",
      })
    }

    setFiles((prev) => [...prev, ...csvFiles])
    setErrorMessage(null)
  }

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index))
  }

  const uploadFiles = async () => {
    if (files.length === 0) return

    setUploadStatus("uploading")
    setUploadProgress(0)
    setErrorMessage(null)

    try {
      // Simulate progress
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

      // Actual upload
      const response = await apiService.uploadCSV(files)

      clearInterval(interval)
      setUploadProgress(100)
      setUploadStatus("success")

      toast({
        title: "Upload Successful",
        description: response.message,
      })

      // Clear files after successful upload
      setTimeout(() => {
        setFiles([])
        setUploadStatus("idle")
        setUploadProgress(null)
      }, 2000)
    } catch (error) {
      console.error("Upload error:", error)
      setUploadStatus("error")
      setErrorMessage(error instanceof Error ? error.message : "Failed to upload files")

      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload files",
        variant: "destructive",
      })
    }
  }

  const triggerProcessing = async () => {
    try {
      setUploadStatus("uploading")
      setUploadProgress(0)
      setErrorMessage(null)

      // Simulate progress
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

      const response = await apiService.triggerProcessing()

      clearInterval(interval)
      setUploadProgress(100)
      setUploadStatus("success")

      toast({
        title: "Processing Triggered",
        description: response.message,
      })

      setTimeout(() => {
        setUploadStatus("idle")
        setUploadProgress(null)
      }, 2000)
    } catch (error) {
      console.error("Processing error:", error)
      setUploadStatus("error")
      setErrorMessage(error instanceof Error ? error.message : "Failed to trigger processing")

      toast({
        title: "Processing Failed",
        description: error instanceof Error ? error.message : "Failed to trigger processing",
        variant: "destructive",
      })
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
        onClick={() => document.getElementById("csv-file-input")?.click()}
      >
        <input
          type="file"
          id="csv-file-input"
          onChange={handleFileInputChange}
          accept=".csv"
          multiple
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center space-y-2">
          <div className="rounded-full bg-primary/10 p-3">
            <Upload className="h-6 w-6 text-primary" />
          </div>
          <div className="text-lg font-medium">
            {uploadStatus === "error" ? "Upload Failed" : "Drop CSV files here or click to upload"}
          </div>
          <div className="text-sm text-muted-foreground">
            Upload your knowledge library CSV files to analyze and enhance
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
          <div className="text-sm font-medium">Selected Files ({files.length}):</div>
          <div className="space-y-2 max-h-60 overflow-y-auto">
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
                  <X className="h-4 w-4" />
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

      <div className="flex justify-center mt-4">
        <Button onClick={triggerProcessing} disabled={uploadStatus === "uploading"} variant="outline">
          Trigger Processing Pipeline
        </Button>
      </div>
    </div>
  )
}
