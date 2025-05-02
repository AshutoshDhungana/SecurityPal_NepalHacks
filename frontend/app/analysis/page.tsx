import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { SimilarityMatrix } from "@/components/similarity-matrix"
import { ClusterVisualization } from "@/components/cluster-visualization"
import { OutdatedEntryList } from "@/components/outdated-entry-list"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"

export default function AnalysisPage() {
  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Analysis Tools</h1>
        <Button>Export Results</Button>
      </div>

      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Analysis Configuration</CardTitle>
            <CardDescription>Adjust parameters for the knowledge library analysis</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="font-medium">Similarity Threshold</div>
                  <div className="text-sm text-muted-foreground">
                    Minimum similarity score to consider entries as similar
                  </div>
                  <Slider defaultValue={[0.75]} max={1} step={0.05} className="py-4" />
                  <div className="text-sm text-muted-foreground">
                    Current: 0.75 (Higher values require more similarity)
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="font-medium">Age Threshold (Days)</div>
                  <div className="text-sm text-muted-foreground">
                    Entries older than this will be flagged for review
                  </div>
                  <Slider defaultValue={[730]} max={1095} step={30} className="py-4" />
                  <div className="text-sm text-muted-foreground">Current: 730 days (2 years)</div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="font-medium">Embedding Model</div>
                  <div className="text-sm text-muted-foreground">Model used for generating text embeddings</div>
                  <Select defaultValue="mini">
                    <SelectTrigger>
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mini">MiniLM (Faster)</SelectItem>
                      <SelectItem value="mpnet">MPNet (More accurate)</SelectItem>
                      <SelectItem value="bert">BERT (Balanced)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <div className="font-medium">Clustering Algorithm</div>
                  <div className="text-sm text-muted-foreground">Algorithm used for grouping similar entries</div>
                  <Select defaultValue="kmeans">
                    <SelectTrigger>
                      <SelectValue placeholder="Select algorithm" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="kmeans">K-Means</SelectItem>
                      <SelectItem value="hierarchical">Hierarchical</SelectItem>
                      <SelectItem value="dbscan">DBSCAN</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button variant="outline">Reset to Defaults</Button>
            <Button>Apply Settings</Button>
          </CardFooter>
        </Card>

        <Tabs defaultValue="similarity">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="similarity">Similarity Analysis</TabsTrigger>
            <TabsTrigger value="clusters">Cluster Visualization</TabsTrigger>
            <TabsTrigger value="outdated">Temporal Analysis</TabsTrigger>
          </TabsList>
          <TabsContent value="similarity" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Entry Similarity Matrix</CardTitle>
                <CardDescription>Heatmap showing similarity between knowledge library entries</CardDescription>
              </CardHeader>
              <CardContent className="h-[500px]">
                <SimilarityMatrix />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="clusters" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Knowledge Clusters</CardTitle>
                <CardDescription>Visual representation of entry clusters based on content similarity</CardDescription>
              </CardHeader>
              <CardContent className="h-[500px]">
                <ClusterVisualization />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="outdated" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Temporal Analysis</CardTitle>
                <CardDescription>Analysis of entry age and potential outdated content</CardDescription>
              </CardHeader>
              <CardContent>
                <OutdatedEntryList />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
