// Types for the knowledge library

export interface KnowledgeEntry {
  product_id: string
  cqid: string
  created_at: string
  category: string
  deleted_at: string | null
  question: string
  answer: string
  details: string
}

export interface SimilarityPair {
  entry1: KnowledgeEntry
  entry2: KnowledgeEntry
  similarity: number
}

export interface Suggestion {
  type: "merge" | "update" | "complete"
  priority: "high" | "medium" | "low"
  entries: KnowledgeEntry[]
  recommendation: string
  similarity?: number
}

export interface AnalysisResult {
  similarityMatrix: number[][]
  similarPairs: SimilarityPair[]
  outdatedEntries: KnowledgeEntry[]
  incompleteEntries: KnowledgeEntry[]
  suggestions: Suggestion[]
}

// Sample data for development and testing
export const sampleKnowledgeLibrary: KnowledgeEntry[] = [
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
]

// Utility functions for working with the knowledge library

// Function to calculate the age of an entry in days
export function calculateEntryAge(entry: KnowledgeEntry, referenceDate: Date = new Date()): number {
  const createdDate = new Date(entry.created_at)
  const ageInMs = referenceDate.getTime() - createdDate.getTime()
  return Math.floor(ageInMs / (1000 * 60 * 60 * 24))
}

// Function to check if an entry is outdated (older than the specified threshold in days)
export function isEntryOutdated(entry: KnowledgeEntry, thresholdDays = 730, referenceDate: Date = new Date()): boolean {
  return calculateEntryAge(entry, referenceDate) > thresholdDays
}

// Function to check if an entry is incomplete (missing answer)
export function isEntryIncomplete(entry: KnowledgeEntry): boolean {
  return !entry.answer || entry.answer.trim() === ""
}
