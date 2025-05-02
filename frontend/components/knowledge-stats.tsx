import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export interface KnowledgeStatsProps {
  totalEntries?: number;
  healthScore?: number;
  clusterCount?: number;
}

export function KnowledgeStats({
  totalEntries = 65,
  healthScore = 75,
  clusterCount = 35
}: KnowledgeStatsProps) {
  const categories = [
    { name: "Information Security", count: Math.round(totalEntries * 0.23), percentage: 23 },
    { name: "Password Policy", count: Math.round(totalEntries * 0.18), percentage: 18 },
    { name: "Data Retention", count: Math.round(totalEntries * 0.15), percentage: 15 },
    { name: "Hosting", count: Math.round(totalEntries * 0.12), percentage: 12 },
    { name: "Organization", count: Math.round(totalEntries * 0.11), percentage: 11 },
    { name: "Other Categories", count: Math.round(totalEntries * 0.21), percentage: 21 },
  ]

  const ages = [
    { range: "< 6 months", count: Math.round(totalEntries * 0.18), percentage: 18 },
    { range: "6-12 months", count: Math.round(totalEntries * 0.23), percentage: 23 },
    { range: "1-2 years", count: Math.round(totalEntries * 0.31), percentage: 31 },
    { range: "2-3 years", count: Math.round(totalEntries * 0.23), percentage: 23 },
    { range: "> 3 years", count: Math.round(totalEntries * 0.05), percentage: 5 },
  ]

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Category Distribution</CardTitle>
          <CardDescription>Breakdown of entries by category</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {categories.map((category) => (
              <div key={category.name} className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="font-medium">{category.name}</div>
                  <div>
                    {category.count} entries ({category.percentage}%)
                  </div>
                </div>
                <div className="h-2 rounded-full bg-muted">
                  <div className="h-full rounded-full bg-primary" style={{ width: `${category.percentage}%` }}></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Age Distribution</CardTitle>
          <CardDescription>Breakdown of entries by age</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {ages.map((age) => (
              <div key={age.range} className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="font-medium">{age.range}</div>
                  <div>
                    {age.count} entries ({age.percentage}%)
                  </div>
                </div>
                <div className="h-2 rounded-full bg-muted">
                  <div
                    className={`h-full rounded-full ${age.range.includes(">")
                        ? "bg-red-500"
                        : age.range.includes("2-3")
                          ? "bg-yellow-500"
                          : "bg-green-500"
                      }`}
                    style={{ width: `${age.percentage}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
