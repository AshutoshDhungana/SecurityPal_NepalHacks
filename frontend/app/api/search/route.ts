import { NextResponse } from "next/server"
import { sampleKnowledgeLibrary } from "@/lib/knowledge-library"

// This would be replaced with actual embedding model in production
// For demo purposes, we're using a simplified approach
function generateEmbedding(text: string) {
  // Simple hash function to simulate embeddings
  let hash = 0
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i)
    hash = (hash << 5) - hash + char
    hash = hash & hash
  }

  // Generate a 5-dimensional "embedding" based on the hash
  const embedding = []
  let h = hash
  for (let i = 0; i < 5; i++) {
    embedding.push((h % 100) / 100) // Normalize to [0, 1]
    h = Math.floor(h / 100)
  }

  return embedding
}

function cosineSimilarity(a: number[], b: number[]) {
  let dotProduct = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }

  normA = Math.sqrt(normA)
  normB = Math.sqrt(normB)

  return dotProduct / (normA * normB)
}

export async function POST(request: Request) {
  try {
    const { query, threshold = 0.7 } = await request.json()

    if (!query) {
      return NextResponse.json({ error: "Query is required" }, { status: 400 })
    }

    // Generate embedding for the query
    const queryEmbedding = generateEmbedding(query.toLowerCase())

    // In a real implementation, this would read from a database
    // For demo purposes, we'll use the sample data
    const entriesWithSimilarity = sampleKnowledgeLibrary.map((entry) => {
      const text = `${entry.question} ${entry.answer} ${entry.details}`.toLowerCase()
      const entryEmbedding = generateEmbedding(text)
      const similarity = cosineSimilarity(queryEmbedding, entryEmbedding)

      return {
        ...entry,
        similarity,
      }
    })

    // Filter by threshold and sort by similarity (highest first)
    const results = entriesWithSimilarity
      .filter((entry) => entry.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)

    return NextResponse.json({ results })
  } catch (error) {
    console.error("Error searching knowledge library:", error)
    return NextResponse.json({ error: "Failed to search knowledge library" }, { status: 500 })
  }
}
