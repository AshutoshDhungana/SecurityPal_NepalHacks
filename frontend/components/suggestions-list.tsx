import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowUpDownIcon as ArrowsUpDownIcon, Merge, Clock, AlertCircle } from "lucide-react"

export function SuggestionsList() {
  const suggestions = [
    {
      id: "suggestion1",
      type: "merge",
      priority: "high",
      title: "Merge similar authentication entries",
      description: "Two entries about multi-factor authentication have high similarity (92%) and should be merged.",
      entries: [
        {
          id: "entry1",
          question: "Are multi-factor authentication capabilities availability?",
          answer: "Yes",
          created: "2020-03-06",
        },
        {
          id: "entry2",
          question: "Does your system support MFA?",
          answer: "Yes, we support TOTP and SMS-based authentication",
          created: "2021-01-15",
        },
      ],
      recommendation: "Merge these entries and keep the more detailed answer from the newer entry.",
    },
    {
      id: "suggestion2",
      type: "update",
      priority: "high",
      title: "Update outdated data retention policy",
      description: "The data retention policy entry is over 3 years old and may not reflect current regulations.",
      entries: [
        {
          id: "entry3",
          question:
            "Do you have documented policies and procedures demonstrating adherence to data retention periods as per legal, statutory or regulatory compliance requirements?",
          answer: "Yes",
          created: "2020-05-18",
        },
      ],
      recommendation:
        "Review and update this entry to ensure compliance with current data retention regulations, including GDPR amendments from 2022.",
    },
    {
      id: "suggestion3",
      type: "complete",
      priority: "medium",
      title: "Complete missing information",
      description: "An entry about financial information is missing an answer.",
      entries: [
        {
          id: "entry4",
          question: "Person Financial Information",
          answer: "",
          created: "2020-02-18",
        },
      ],
      recommendation: "Add a complete answer to this entry or consider removing it if no longer relevant.",
    },
  ]

  const getIconForType = (type: string) => {
    switch (type) {
      case "merge":
        return <Merge className="h-4 w-4" />
      case "update":
        return <Clock className="h-4 w-4" />
      case "complete":
        return <AlertCircle className="h-4 w-4" />
      default:
        return <ArrowsUpDownIcon className="h-4 w-4" />
    }
  }

  const getColorForPriority = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-100 text-red-800"
      case "medium":
        return "bg-yellow-100 text-yellow-800"
      case "low":
        return "bg-blue-100 text-blue-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="space-y-6">
      {suggestions.map((suggestion) => (
        <Card key={suggestion.id} className="p-4">
          <div className="flex items-start gap-4">
            <div
              className={`flex h-8 w-8 items-center justify-center rounded-full ${getColorForPriority(suggestion.priority)}`}
            >
              {getIconForType(suggestion.type)}
            </div>

            <div className="flex-1 space-y-3">
              <div className="flex items-center justify-between">
                <div className="font-medium">{suggestion.title}</div>
                <Badge
                  variant={
                    suggestion.priority === "high"
                      ? "destructive"
                      : suggestion.priority === "medium"
                        ? "default"
                        : "secondary"
                  }
                >
                  {suggestion.priority} priority
                </Badge>
              </div>

              <div className="text-sm text-muted-foreground">{suggestion.description}</div>

              <div className="space-y-2 rounded-md border p-3">
                <div className="text-xs font-medium uppercase text-muted-foreground">Affected Entries</div>

                {suggestion.entries.map((entry) => (
                  <div key={entry.id} className="space-y-1 rounded-md bg-muted/50 p-2 text-sm">
                    <div className="font-medium">{entry.question}</div>
                    <div className="text-muted-foreground">Answer: {entry.answer || "(Missing)"}</div>
                    <div className="text-xs text-muted-foreground">Created: {entry.created}</div>
                  </div>
                ))}
              </div>

              <div className="rounded-md bg-blue-50 p-2 text-sm text-blue-800">
                <div className="font-medium">Recommendation:</div>
                <div>{suggestion.recommendation}</div>
              </div>

              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm">
                  Ignore
                </Button>
                <Button size="sm">Implement</Button>
              </div>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}
