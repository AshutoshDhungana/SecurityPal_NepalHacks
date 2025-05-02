import { NextResponse } from "next/server"

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
    const { threshold = 0.75 } = await request.json()

    // In a real implementation, this would read from a database
    // For demo purposes, we'll use the sample data
    const sampleData = [
      {
        product_id: "47a5a380-9214-4db3-b372-0c358d1350a1",
        cqid: "454d9589-233b-4277-8d60-f9782efc3595",
        created_at: "2020-05-18 00:00:00",
        category: "Data Retention and Deletion",
        deleted_at: null,
        question:
          "Do you have documented policies and procedures demonstrating adherence to data retention periods as per legal, statutory or regulatory compliance requirements?",
        answer: "Yes",
        details: "",
      },
      {
        product_id: "47a5a380-9214-4db3-b372-0c358d1350a1",
        cqid: "6fb817a8-ad90-4e0d-bddd-5ec209335b52",
        created_at: "2020-02-27 00:00:00",
        category: "Hosting",
        deleted_at: null,
        question: "Is the application provided as a Cloud service? (Public, Private, Hybrid)",
        answer: "Yes",
        details: "Danfe Corp uses Amazon Web Services",
      },
      {
        product_id: "47a5a380-9214-4db3-b372-0c358d1350a1",
        cqid: "cf87847a-7810-43af-9453-a23c047eab51",
        created_at: "2020-02-18 00:00:00",
        category: "Organization",
        deleted_at: null,
        question: "Person Financial Information",
        answer: "",
        details: "",
      },
      {
        product_id: "47a5a380-9214-4db3-b372-0c358d1350a1",
        cqid: "b0807ac1-8672-466b-8bd8-0f656b01b8ec",
        created_at: "2020-03-06 00:00:00",
        category: "Password Policy and authentication procedures",
        deleted_at: null,
        question: "Are multi-factor authentication capabilities availability?",
        answer: "Yes",
        details: "",
      },
      {
        product_id: "47a5a380-9214-4db3-b372-0c358d1350a1",
        cqid: "10f57b4d-1bb0-4898-90cf-18a30c3ad615",
        created_at: "2019-11-28 00:00:00",
        category: "Information Security",
        deleted_at: null,
        question: "Information security policy",
        answer: "Yes",
        details: "Yes, certified by SOC 2",
      },
      // Additional entries for demonstration
      {
        product_id: "47a5a380-9214-4db3-b372-0c358d1350a1",
        cqid: "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
        created_at: "2021-01-15 00:00:00",
        category: "Password Policy and authentication procedures",
        deleted_at: null,
        question: "Does your system support MFA?",
        answer: "Yes, we support TOTP and SMS-based authentication",
        details: "",
      },
      {
        product_id: "47a5a380-9214-4db3-b372-0c358d1350a1",
        cqid: "q1r2s3t4-u5v6-w7x8-y9z0-a1b2c3d4e5f6",
        created_at: "2021-04-10 00:00:00",
        category: "Information Security",
        deleted_at: null,
        question: "Do you have a documented information security policy?",
        answer: "Yes, we maintain ISO 27001 certification",
        details: "",
      },
    ]

    // Generate embeddings for each entry
    const entriesWithEmbeddings = sampleData.map((entry) => {
      const text = `${entry.question} ${entry.answer} ${entry.details}`.toLowerCase()
      const embedding = generateEmbedding(text)

      return {
        ...entry,
        embedding,
      }
    })

    // Calculate similarity matrix
    const similarityMatrix = []
    const similarPairs = []

    for (let i = 0; i < entriesWithEmbeddings.length; i++) {
      similarityMatrix[i] = []

      for (let j = 0; j < entriesWithEmbeddings.length; j++) {
        const similarity = cosineSimilarity(entriesWithEmbeddings[i].embedding, entriesWithEmbeddings[j].embedding)

        similarityMatrix[i][j] = similarity

        // Identify similar pairs (excluding self-comparisons)
        if (i !== j && similarity >= threshold) {
          similarPairs.push({
            entry1: entriesWithEmbeddings[i],
            entry2: entriesWithEmbeddings[j],
            similarity,
          })
        }
      }
    }

    // Identify outdated entries (older than 2 years from a reference date)
    const referenceDate = new Date("2023-01-01") // Using a fixed reference date for demo
    const outdatedThreshold = 2 * 365 * 24 * 60 * 60 * 1000 // 2 years in milliseconds

    const outdatedEntries = entriesWithEmbeddings.filter((entry) => {
      const createdDate = new Date(entry.created_at)
      const age = referenceDate.getTime() - createdDate.getTime()
      return age > outdatedThreshold
    })

    // Identify incomplete entries
    const incompleteEntries = entriesWithEmbeddings.filter((entry) => {
      return !entry.answer || entry.answer.trim() === ""
    })

    // Generate suggestions
    const suggestions = []

    // Suggestions for similar entries
    similarPairs.forEach((pair) => {
      suggestions.push({
        type: "merge",
        priority: pair.similarity > 0.9 ? "high" : "medium",
        entries: [pair.entry1, pair.entry2],
        similarity: pair.similarity,
        recommendation: `Merge these similar entries and keep the more detailed or recent information.`,
      })
    })

    // Suggestions for outdated entries
    outdatedEntries.forEach((entry) => {
      suggestions.push({
        type: "update",
        priority: "high",
        entries: [entry],
        recommendation: `This entry is over 2 years old and may contain outdated information. Review and update as needed.`,
      })
    })

    // Suggestions for incomplete entries
    incompleteEntries.forEach((entry) => {
      suggestions.push({
        type: "complete",
        priority: "medium",
        entries: [entry],
        recommendation: `This entry is missing an answer. Complete the information or remove if no longer relevant.`,
      })
    })

    return NextResponse.json({
      similarityMatrix,
      similarPairs,
      outdatedEntries,
      incompleteEntries,
      suggestions,
    })
  } catch (error) {
    console.error("Error analyzing knowledge library:", error)
    return NextResponse.json({ error: "Failed to analyze knowledge library" }, { status: 500 })
  }
}
