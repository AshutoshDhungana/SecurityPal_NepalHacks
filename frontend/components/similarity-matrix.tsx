"use client"

import { useEffect, useRef } from "react"

export function SimilarityMatrix() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    // Sample data - similarity matrix
    const entries = [
      "Data Retention Policy",
      "MFA Capabilities",
      "Information Security Policy",
      "Cloud Service",
      "Person Financial Info",
      "MFA Support",
      "ISO 27001 Certification",
      "AWS Hosting",
      "Data Privacy Policy",
      "Password Requirements",
    ]

    // Generate a sample similarity matrix (values between 0 and 1)
    const similarityMatrix = Array(entries.length)
      .fill(0)
      .map(() =>
        Array(entries.length)
          .fill(0)
          .map(() => Math.random()),
      )

    // Set diagonal to 1 (self-similarity)
    for (let i = 0; i < entries.length; i++) {
      similarityMatrix[i][i] = 1
    }

    // Set some high similarities for related entries
    similarityMatrix[1][5] = 0.92 // MFA Capabilities and MFA Support
    similarityMatrix[5][1] = 0.92

    similarityMatrix[2][6] = 0.85 // Information Security Policy and ISO 27001
    similarityMatrix[6][2] = 0.85

    similarityMatrix[3][7] = 0.88 // Cloud Service and AWS Hosting
    similarityMatrix[7][3] = 0.88

    similarityMatrix[0][8] = 0.78 // Data Retention and Data Privacy
    similarityMatrix[8][0] = 0.78

    // Chart dimensions
    const padding = 150
    const cellSize = Math.min(
      (canvas.width - padding * 2) / entries.length,
      (canvas.height - padding * 2) / entries.length,
    )
    const chartSize = cellSize * entries.length
    const startX = (canvas.width - chartSize) / 2
    const startY = (canvas.height - chartSize) / 2

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw cells
    for (let i = 0; i < entries.length; i++) {
      for (let j = 0; j < entries.length; j++) {
        const similarity = similarityMatrix[i][j]

        // Color based on similarity (red for low, green for high)
        const r = Math.floor(255 * (1 - similarity))
        const g = Math.floor(255 * similarity)
        const b = 0

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
        ctx.fillRect(startX + j * cellSize, startY + i * cellSize, cellSize, cellSize)

        // Add similarity value text for cells with high similarity
        if (similarity > 0.7 && i !== j) {
          ctx.fillStyle = "white"
          ctx.font = "bold 10px sans-serif"
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText(
            similarity.toFixed(2),
            startX + j * cellSize + cellSize / 2,
            startY + i * cellSize + cellSize / 2,
          )
        }
      }
    }

    // Draw row labels
    ctx.fillStyle = "#334155"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"

    for (let i = 0; i < entries.length; i++) {
      ctx.fillText(entries[i], startX - 10, startY + i * cellSize + cellSize / 2)
    }

    // Draw column labels
    ctx.textAlign = "left"
    ctx.textBaseline = "top"

    for (let j = 0; j < entries.length; j++) {
      ctx.save()
      ctx.translate(startX + j * cellSize + cellSize / 2, startY - 10)
      ctx.rotate(-Math.PI / 4)
      ctx.fillText(entries[j], 0, 0)
      ctx.restore()
    }

    // Draw legend
    const legendWidth = 200
    const legendHeight = 20
    const legendX = (canvas.width - legendWidth) / 2
    const legendY = canvas.height - 40

    // Create gradient
    const gradient = ctx.createLinearGradient(legendX, 0, legendX + legendWidth, 0)
    gradient.addColorStop(0, "rgb(255, 0, 0)") // Low similarity
    gradient.addColorStop(1, "rgb(0, 255, 0)") // High similarity

    ctx.fillStyle = gradient
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight)

    // Legend labels
    ctx.fillStyle = "#334155"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"

    ctx.fillText("Low Similarity", legendX, legendY + legendHeight + 5)
    ctx.fillText("High Similarity", legendX + legendWidth, legendY + legendHeight + 5)
  }, [])

  return (
    <div className="w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full"></canvas>
    </div>
  )
}
