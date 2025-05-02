"use client"

import { useEffect, useRef } from "react"

export function ClusterVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    // Sample data - clusters of entries
    const clusters = [
      {
        name: "Authentication",
        color: "#3b82f6",
        entries: [
          { id: "entry1", name: "MFA Capabilities", x: 0.3, y: 0.2 },
          { id: "entry2", name: "MFA Support", x: 0.35, y: 0.25 },
          { id: "entry3", name: "Password Requirements", x: 0.4, y: 0.3 },
          { id: "entry4", name: "Login Security", x: 0.25, y: 0.15 },
        ],
      },
      {
        name: "Information Security",
        color: "#ef4444",
        entries: [
          { id: "entry5", name: "Information Security Policy", x: 0.7, y: 0.2 },
          { id: "entry6", name: "ISO 27001 Certification", x: 0.75, y: 0.25 },
          { id: "entry7", name: "Security Controls", x: 0.8, y: 0.3 },
        ],
      },
      {
        name: "Data Privacy",
        color: "#22c55e",
        entries: [
          { id: "entry8", name: "Data Retention Policy", x: 0.3, y: 0.7 },
          { id: "entry9", name: "Data Privacy Policy", x: 0.35, y: 0.75 },
          { id: "entry10", name: "GDPR Compliance", x: 0.4, y: 0.8 },
        ],
      },
      {
        name: "Hosting",
        color: "#f97316",
        entries: [
          { id: "entry11", name: "Cloud Service", x: 0.7, y: 0.7 },
          { id: "entry12", name: "AWS Hosting", x: 0.75, y: 0.75 },
        ],
      },
    ]

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw connections within clusters
    clusters.forEach((cluster) => {
      ctx.strokeStyle = cluster.color
      ctx.lineWidth = 1

      // Connect all entries in the cluster
      for (let i = 0; i < cluster.entries.length; i++) {
        for (let j = i + 1; j < cluster.entries.length; j++) {
          const entry1 = cluster.entries[i]
          const entry2 = cluster.entries[j]

          ctx.beginPath()
          ctx.moveTo(entry1.x * canvas.width, entry1.y * canvas.height)
          ctx.lineTo(entry2.x * canvas.width, entry2.y * canvas.height)
          ctx.stroke()
        }
      }
    })

    // Draw nodes
    clusters.forEach((cluster) => {
      cluster.entries.forEach((entry) => {
        // Draw circle
        ctx.beginPath()
        ctx.fillStyle = cluster.color
        ctx.arc(entry.x * canvas.width, entry.y * canvas.height, 8, 0, Math.PI * 2)
        ctx.fill()

        // Draw label
        ctx.fillStyle = "#334155"
        ctx.font = "12px sans-serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "top"
        ctx.fillText(entry.name, entry.x * canvas.width, entry.y * canvas.height + 12)
      })
    })

    // Draw cluster labels
    clusters.forEach((cluster) => {
      // Calculate cluster center
      const centerX = cluster.entries.reduce((sum, entry) => sum + entry.x, 0) / cluster.entries.length
      const centerY = cluster.entries.reduce((sum, entry) => sum + entry.y, 0) / cluster.entries.length

      // Draw label background
      const labelText = `${cluster.name} (${cluster.entries.length})`
      const textWidth = ctx.measureText(labelText).width

      ctx.fillStyle = "rgba(255, 255, 255, 0.8)"
      ctx.fillRect(centerX * canvas.width - textWidth / 2 - 5, centerY * canvas.height - 25, textWidth + 10, 20)

      // Draw label text
      ctx.fillStyle = cluster.color
      ctx.font = "bold 14px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(labelText, centerX * canvas.width, centerY * canvas.height - 15)
    })

    // Draw legend
    const legendX = 20
    let legendY = 20
    const legendSpacing = 25

    clusters.forEach((cluster) => {
      // Draw color circle
      ctx.beginPath()
      ctx.fillStyle = cluster.color
      ctx.arc(legendX, legendY, 8, 0, Math.PI * 2)
      ctx.fill()

      // Draw label
      ctx.fillStyle = "#334155"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"
      ctx.fillText(`${cluster.name} (${cluster.entries.length})`, legendX + 15, legendY)

      legendY += legendSpacing
    })
  }, [])

  return (
    <div className="w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full"></canvas>
    </div>
  )
}
