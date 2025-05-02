import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CSVUploader } from "@/components/csv-uploader"

export default function ImportPage() {
  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Import Data</h1>
        <Button>View Import History</Button>
      </div>

      <Tabs defaultValue="knowledge-library">
        <TabsList className="grid w-full grid-cols-1 mb-6">
          <TabsTrigger value="knowledge-library">Knowledge Library</TabsTrigger>
        </TabsList>

        <TabsContent value="knowledge-library">
          <Card>
            <CardHeader>
              <CardTitle>Import Knowledge Library</CardTitle>
              <CardDescription>
                Upload your security Q&A pairs in CSV format to analyze and enhance your knowledge library
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CSVUploader />
            </CardContent>
            <CardFooter className="flex flex-col items-start gap-2">
              <div className="text-sm font-medium">Expected CSV Format:</div>
              <div className="text-sm text-muted-foreground">
                Your CSV should include columns for: product_id, cqid, created_at, category, question, answer, details
              </div>
              <Button variant="outline" className="mt-2">
                Download Template
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
