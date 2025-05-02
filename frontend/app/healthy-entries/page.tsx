import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { CheckCircle2 } from "lucide-react"

export default function HealthyEntriesPage() {
  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Healthy Entries</h1>
        <Button>Export List</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Healthy Entries</CardTitle>
          <CardDescription>Up-to-date and complete entries that do not require attention</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="rounded-md border p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-medium">System Access Control</div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Answer: Access is controlled through role-based permissions and least privilege principles
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Created: January 3, 2023</div>
                </div>
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-100 text-green-500">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
              </div>
            </div>

            <div className="rounded-md border p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-medium">Encryption Standards</div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Answer: All data is encrypted at rest using AES-256 and in transit using TLS 1.3
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Created: February 15, 2023</div>
                </div>
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-100 text-green-500">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
              </div>
            </div>

            <div className="rounded-md border p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-medium">Application Security Testing</div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Answer: We perform regular SAST, DAST, and penetration testing of all applications
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Created: March 22, 2023</div>
                </div>
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-100 text-green-500">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
              </div>
            </div>

            <div className="rounded-md border p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-medium">Vulnerability Management</div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Answer: We have automated vulnerability scanning and a defined patch management process
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Created: April 7, 2023</div>
                </div>
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-100 text-green-500">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
              </div>
            </div>

            <div className="text-center mt-4">
              <Button variant="outline">Load More</Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
