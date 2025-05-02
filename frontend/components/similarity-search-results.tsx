import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

interface SearchResult {
  id: string
  question: string
  answer: string
  category: string
  created_at: string
  similarity: number
}

interface SimilaritySearchResultsProps {
  results: SearchResult[]
  query: string
}

export function SimilaritySearchResults({ results, query }: SimilaritySearchResultsProps) {
  // Function to format similarity score as percentage
  const formatSimilarity = (score: number) => {
    return `${(score * 100).toFixed(1)}%`
  }

  // Function to determine badge color based on similarity score
  const getBadgeVariant = (score: number) => {
    if (score >= 0.9) return "default"
    if (score >= 0.8) return "secondary"
    return "outline"
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Search Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4 p-3 bg-muted/50 rounded-md">
          <div className="text-sm font-medium">Your query:</div>
          <div className="text-sm mt-1">{query}</div>
        </div>

        {results.length === 0 ? (
          <div className="text-center py-8">
            <div className="text-lg font-medium">No similar questions found</div>
            <div className="text-sm text-muted-foreground mt-1">
              Try adjusting your search query or lowering the similarity threshold
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {results.map((result) => (
              <div key={result.id} className="rounded-md border p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="font-medium">{result.question}</div>
                    <div className="text-sm text-muted-foreground mt-1">
                      Answer: {result.answer || "(No answer provided)"}
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <div className="text-xs text-muted-foreground">Category: {result.category}</div>
                      <div className="text-xs text-muted-foreground">
                        Created: {new Date(result.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                  <Badge variant={getBadgeVariant(result.similarity)}>
                    {formatSimilarity(result.similarity)} similar
                  </Badge>
                </div>
                <div className="flex justify-end gap-2 mt-4">
                  <Button variant="outline" size="sm">
                    View Details
                  </Button>
                  <Button size="sm">Use This Answer</Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
