"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { apiService, type Cluster } from "@/lib/api-service"
import { useToast } from "@/hooks/use-toast"

export default function SimilarEntriesPage() {
  const [clusters, setClusters] = useState<Cluster[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { toast } = useToast()

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null) // Reset any previous errors
        const results = await apiService.getResults()

        // Filter to only include clusters with multiple entries
        const validClusters = results.filter(
          (cluster: any) => cluster.entries && Array.isArray(cluster.entries) && cluster.entries.length > 1,
        )

        setClusters(validClusters as Cluster[])
      } catch (err) {
        console.error("Error fetching data:", err)
        setError(err instanceof Error ? err.message : "Failed to fetch data")
        toast({
          title: "Warning",
          description: "Using mock data - backend connection failed",
          variant: "warning",
        })
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [toast])

  const handleMergeEntries = async (cluster: Cluster) => {
    try {
      const cqids = cluster.entries.map((entry) => entry.canonicalid)
      await apiService.mergeQuestions(cqids)

      toast({
        title: "Success",
        description: "Entries merged successfully",
      })

      // Remove the cluster from the list
      setClusters(clusters.filter((c) => c.cluster_id !== cluster.cluster_id))
    } catch (err) {
      toast({
        title: "Error",
        description: err instanceof Error ? err.message : "Failed to merge entries",
        variant: "destructive",
      })
    }
  }

  if (loading) {
    return (
      <div className="p-6 flex flex-col items-center justify-center min-h-[60vh]">
        <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
        <p className="mt-4 text-lg">Loading similar entries...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center justify-center text-center p-6">
              <div className="text-red-500 mb-4">⚠️</div>
              <h2 className="text-2xl font-bold mb-2">Error Loading Data</h2>
              <p className="text-muted-foreground mb-4">{error}</p>
              <Button onClick={() => window.location.reload()}>Retry</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (clusters.length === 0) {
    return (
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Similar Entries</h1>
          <Button>Export List</Button>
        </div>

        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center justify-center text-center p-6">
              <p className="text-lg mb-4">No similar entries found</p>
              <Button onClick={() => window.location.reload()}>Refresh</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Similar Entries</h1>
        <Button>Export List</Button>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Similar Entry Clusters</CardTitle>
          <CardDescription>Groups of entries with similar content that could be merged</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-8">
            {clusters.map((cluster) => (
              <div key={cluster.cluster_id} className="space-y-4">
                <div className="font-medium">{cluster.category} Cluster</div>
                <div className="rounded-md border">
                  {cluster.entries.map((entry, index) => (
                    <div
                      key={entry.canonicalid}
                      className={`p-4 ${index < cluster.entries.length - 1 ? "border-b" : ""} ${index % 2 === 1 ? "bg-muted/50" : ""}`}
                    >
                      <div className="font-medium">{entry.question}</div>
                      <div className="text-sm text-muted-foreground mt-1">Answer: {entry.answer || "(Missing)"}</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Created: {new Date(entry.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  ))}
                  <div className="p-4 flex justify-end gap-2">
                    <Button variant="outline" size="sm">
                      Ignore
                    </Button>
                    <Button size="sm" onClick={() => handleMergeEntries(cluster)}>
                      Merge Entries
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
