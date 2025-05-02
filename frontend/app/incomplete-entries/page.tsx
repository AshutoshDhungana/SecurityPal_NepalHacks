import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { AlertCircle } from "lucide-react"

export default function IncompleteEntriesPage() {
  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Incomplete Entries</h1>
        <Button>Export List</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Incomplete Entries</CardTitle>
          <CardDescription>Entries with missing information that need to be completed</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="rounded-md border p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-medium">Person Financial Information</div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Answer: <span className="text-red-500">(Missing)</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Created: February 18, 2020</div>
                </div>
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-yellow-100 text-yellow-500">
                  <AlertCircle className="h-4 w-4" />
                </div>
              </div>
              <div className="mt-4">
                <div className="text-sm font-medium">Reason for flagging:</div>
                <div className="text-sm text-muted-foreground">
                  This entry is missing an answer. It should be completed with information on how personal financial
                  information is handled.
                </div>
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <Button variant="outline" size="sm">
                  Remove Entry
                </Button>
                <Button size="sm">Add Answer</Button>
              </div>
            </div>

            <div className="rounded-md border p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-medium">Backup and Disaster Recovery</div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Answer: <span className="text-red-500">(Missing)</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Created: March 10, 2021</div>
                </div>
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-yellow-100 text-yellow-500">
                  <AlertCircle className="h-4 w-4" />
                </div>
              </div>
              <div className="mt-4">
                <div className="text-sm font-medium">Reason for flagging:</div>
                <div className="text-sm text-muted-foreground">
                  This entry is missing details about backup procedures and disaster recovery plans.
                </div>
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <Button variant="outline" size="sm">
                  Remove Entry
                </Button>
                <Button size="sm">Add Answer</Button>
              </div>
            </div>

            <div className="rounded-md border p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-medium">Third-Party Risk Management</div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Answer: <span className="text-red-500">(Missing)</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Created: June 5, 2021</div>
                </div>
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-yellow-100 text-yellow-500">
                  <AlertCircle className="h-4 w-4" />
                </div>
              </div>
              <div className="mt-4">
                <div className="text-sm font-medium">Reason for flagging:</div>
                <div className="text-sm text-muted-foreground">
                  This entry needs information about how third-party risks are assessed and managed.
                </div>
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <Button variant="outline" size="sm">
                  Remove Entry
                </Button>
                <Button size="sm">Add Answer</Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
