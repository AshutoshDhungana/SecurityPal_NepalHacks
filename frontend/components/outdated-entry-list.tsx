"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Clock, AlertTriangle } from "lucide-react"
import { apiService } from "@/lib/api-service"
import { useToast } from "@/hooks/use-toast"

interface OutdatedEntry {
  id: string
  question: string
  answer: string
  created_at: string
  age: number
  category: string
  reason: string
}

export function OutdatedEntryList() {
  const [outdatedEntries, setOutdatedEntries] = useState<OutdatedEntry[]>([])
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const results = await apiService.getResults()

        // Process results to find outdated entries
        const now = new Date()
        const outdated = []

        for (const cluster of results) {
          if (cluster.entries && Array.isArray(cluster.entries)) {
            for (const entry of cluster.entries) {
              const createdDate = new Date(entry.created_at)
              const ageInDays = Math.floor((now.getTime() - createdDate.getTime()) / (1000 * 60 * 60 * 24))

              if (ageInDays > 730) {
                // Older than 2 years
                outdated.push({
                  id: entry.id || entry.canonicalid,
                  question: entry.question,
                  answer: entry.answer || "",
                  created_at: entry.created_at,
                  age: ageInDays,
                  category: entry.category || "Unknown",
                  reason: getOutdatedReason(entry.category, ageInDays),
                })
              }
            }
          }
        }

        setOutdatedEntries(outdated)
      } catch (err) {
        console.error("Error fetching data:", err)
        toast({
          title: "Warning",
          description: "Using mock data - backend connection failed",
          variant: "warning",
        })

        // Set some mock outdated entries
        setOutdatedEntries([
          {
            id: "10f57b4d-1bb0-4898-90cf-18a30c3ad615",
            question: "Information security policy",
            answer: "Yes, certified by SOC 2",
            created_at: "2019-11-28 00:00:00",
            age: 1200,
            category: "Information Security",
            reason:
              "Security certification typically requires annual renewal. This entry is over 2 years old and may not reflect current certification status.",
          },
        ])
      } finally {
        setLoading(false)
      }
    }

    // Helper function to generate reason based on category and age
    const getOutdatedReason = (category: string, age: number): string => {
      const categories: Record<string, string> = {
        "Data Retention and Deletion":
          "Data retention regulations have been updated since this entry was created. GDPR amendments in 2022 and CCPA updates may affect this answer.",
        Hosting:
          "Cloud infrastructure information is likely outdated. AWS offerings have changed significantly since 2020, and the company may have adopted a multi-cloud strategy.",
        "Information Security":
          "Security certification typically requires annual renewal. This entry is over 2 years old and may not reflect current certification status.",
        "Password Policy and authentication procedures":
          "Authentication standards have evolved. This entry lacks details on the types of MFA supported and may not reflect current capabilities.",
      }

      return (
        categories[category] || `This entry is ${age} days old and may contain outdated information that needs review.`
      )
    }

    fetchData()
  }, [toast])

  const handleUpdateEntry = async (entryId: string) => {
    try {
      // In a real implementation, you would navigate to an edit page or open a modal
      toast({
        title: "Update Entry",
        description: `Navigating to edit page for entry ${entryId}`,
      })
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to update entry",
        variant: "destructive",
      })
    }
  }

  const handleMarkAsCurrent = async (entryId: string) => {
    try {
      // In a real implementation, you would update the entry's timestamp
      toast({
        title: "Marked as Current",
        description: "Entry has been marked as current",
      })

      // Remove from the list
      setOutdatedEntries(outdatedEntries.filter((entry) => entry.id !== entryId))
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to mark entry as current",
        variant: "destructive",
      })
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
      </div>
    )
  }

  if (outdatedEntries.length === 0) {
    return (
      <div className="text-center p-8">
        <p className="text-muted-foreground">No outdated entries found</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          Showing {outdatedEntries.length} potentially outdated entries (older than 730 days)
        </div>
        <Button variant="outline" size="sm">
          Export List
        </Button>
      </div>

      {outdatedEntries.map((entry) => (
        <Card key={entry.id} className="p-4">
          <div className="flex items-start gap-4">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-red-100 text-red-500">
              {entry.age > 1095 ? <AlertTriangle className="h-4 w-4" /> : <Clock className="h-4 w-4" />}
            </div>

            <div className="flex-1 space-y-2">
              <div>
                <div className="font-medium">{entry.question}</div>
                <div className="text-sm text-muted-foreground">Answer: {entry.answer}</div>
              </div>

              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
                <div className="text-muted-foreground">
                  <span className="font-medium">Created:</span> {new Date(entry.created_at).toLocaleDateString()}
                </div>
                <div className="text-muted-foreground">
                  <span className="font-medium">Age:</span> {entry.age} days
                </div>
                <div className="text-muted-foreground">
                  <span className="font-medium">Category:</span> {entry.category}
                </div>
              </div>

              <div className="rounded-md bg-amber-50 p-2 text-sm text-amber-800">
                <div className="font-medium">Reason for flagging:</div>
                <div>{entry.reason}</div>
              </div>

              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => handleMarkAsCurrent(entry.id)}>
                  Mark as Current
                </Button>
                <Button size="sm" onClick={() => handleUpdateEntry(entry.id)}>
                  Update Entry
                </Button>
              </div>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}
