import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { OutdatedEntryList } from "@/components/outdated-entry-list"

export default function OutdatedEntriesPage() {
  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Outdated Entries</h1>
        <Button>Export List</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Outdated Entry Analysis</CardTitle>
          <CardDescription>Entries that may contain outdated information</CardDescription>
        </CardHeader>
        <CardContent>
          <OutdatedEntryList />
        </CardContent>
      </Card>
    </div>
  )
}
