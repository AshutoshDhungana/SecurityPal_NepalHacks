//@ts-nocheck

'use client';

import { useState, useEffect } from 'react';
import MainLayout from '@/components/MainLayout';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CSVUploader } from "@/components/csv-uploader";
import { apiService } from "@/lib/api-service";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from '@/components/ui/separator';
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { CheckCircle, AlertCircle, Clock, FilePlus, RefreshCw } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';

export default function ImportPage() {
  const [importHistory, setImportHistory] = useState([]);
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState('');
  const [processType, setProcessType] = useState('all');
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [processingDialogOpen, setProcessingDialogOpen] = useState(false);

  // Fetch import history and products when component mounts
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch import history
        setHistoryLoading(true);
        const history = await apiService.getImportHistory();
        if (history) {
          setImportHistory(history);
        }

        // Fetch available products
        const productsData = await apiService.getProducts();
        if (productsData) {
          setProducts(productsData);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        toast.error('Failed to load import history');
      } finally {
        setHistoryLoading(false);
      }
    };

    fetchData();
  }, []);

  // Handle triggering the processing pipeline
  const handleTriggerProcessing = async () => {
    setLoading(true);
    try {
      const options = {
        productId: selectedProduct || undefined,
        processType: processType as 'embedding' | 'clustering' | 'all'
      };

      const result = await apiService.triggerProcessing(options);

      if (result.success) {
        toast.success(result.message || 'Processing started successfully');
      } else {
        toast.error(result.message || 'Failed to start processing');
      }

      setProcessingDialogOpen(false);
    } catch (error) {
      console.error('Error triggering processing:', error);
      toast.error('Failed to trigger processing');
    } finally {
      setLoading(false);
    }
  };

  // Handle uploading complete notification (to refresh history)
  const handleUploadComplete = async () => {
    try {
      const history = await apiService.getImportHistory();
      if (history) {
        setImportHistory(history);
      }
    } catch (error) {
      console.error('Error refreshing import history:', error);
    }
  };

  // Format date from ISO string
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  // Get status badge style based on status
  const getStatusBadge = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
          <CheckCircle className="w-3 h-3 mr-1" /> Completed
        </span>;
      case 'processing':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
          <RefreshCw className="w-3 h-3 mr-1 animate-spin" /> Processing
        </span>;
      case 'failed':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
          <AlertCircle className="w-3 h-3 mr-1" /> Failed
        </span>;
      case 'pending':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
          <Clock className="w-3 h-3 mr-1" /> Pending
        </span>;
      default:
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
          {status || 'Unknown'}
        </span>;
    }
  };

  return (
    <MainLayout>
      <div className="container mx-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-blue-600">Import Dataset</h1>

          <Dialog open={processingDialogOpen} onOpenChange={setProcessingDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <FilePlus className="mr-2 h-4 w-4" />
                Process Imported Data
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Start Processing Pipeline</DialogTitle>
                <DialogDescription>
                  Select a product to process only that data, or leave empty to process all products.
                </DialogDescription>
              </DialogHeader>

              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <label htmlFor="product" className="text-right font-medium">
                    Product:
                  </label>
                  <Select value={selectedProduct} onValueChange={(value) => setSelectedProduct(value)}>
                    <SelectTrigger className="col-span-3">
                      <SelectValue placeholder="All Products" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All Products</SelectItem>
                      {products.map((product) => (
                        <SelectItem key={product.product_id} value={product.product_id}>
                          {product.product_name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <label htmlFor="process-type" className="text-right font-medium">
                    Process Type:
                  </label>
                  <Select value={processType} onValueChange={(value) => setProcessType(value)}>
                    <SelectTrigger className="col-span-3">
                      <SelectValue placeholder="Full Process" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Full Process</SelectItem>
                      <SelectItem value="embedding">Generate Embeddings Only</SelectItem>
                      <SelectItem value="clustering">Clustering Only</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <DialogFooter>
                <Button variant="outline" onClick={() => setProcessingDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleTriggerProcessing} disabled={loading}>
                  {loading ? (
                    <><RefreshCw className="mr-2 h-4 w-4 animate-spin" /> Processing...</>
                  ) : (
                    <>Start Processing</>
                  )}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>

        <Tabs defaultValue="upload">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="upload">Upload Dataset</TabsTrigger>
            <TabsTrigger value="history">Import History</TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Upload Knowledge Library Dataset</CardTitle>
                    <CardDescription>
                      Upload your Q&A pairs in CSV format to analyze and enhance your knowledge library
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CSVUploader onUploadComplete={handleUploadComplete} />
                  </CardContent>
                  <CardFooter className="flex flex-col items-start gap-2 border-t pt-4">
                    <div className="text-sm font-medium">Expected CSV Format:</div>
                    <div className="text-sm text-muted-foreground">
                      Your CSV should include the following columns:
                    </div>
                    <table className="w-full text-xs my-2">
                      <thead>
                        <tr className="border-b">
                          <th className="py-1 text-left font-medium">Column</th>
                          <th className="py-1 text-left font-medium">Description</th>
                          <th className="py-1 text-left font-medium">Required</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b">
                          <td className="py-1 font-medium">product_id</td>
                          <td className="py-1">Unique identifier for the product</td>
                          <td className="py-1 text-green-600">Yes</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-1 font-medium">cqid</td>
                          <td className="py-1">Canonical question ID</td>
                          <td className="py-1 text-green-600">Yes</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-1 font-medium">question</td>
                          <td className="py-1">The question text</td>
                          <td className="py-1 text-green-600">Yes</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-1 font-medium">answer</td>
                          <td className="py-1">The answer text</td>
                          <td className="py-1 text-green-600">Yes</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-1 font-medium">category</td>
                          <td className="py-1">Question category</td>
                          <td className="py-1 text-yellow-600">No</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-1 font-medium">created_at</td>
                          <td className="py-1">Creation date (YYYY-MM-DD)</td>
                          <td className="py-1 text-yellow-600">No</td>
                        </tr>
                        <tr>
                          <td className="py-1 font-medium">details</td>
                          <td className="py-1">Additional details</td>
                          <td className="py-1 text-yellow-600">No</td>
                        </tr>
                      </tbody>
                    </table>
                    <Button variant="outline" className="mt-2" onClick={() => {
                      // This would download a template file in a real application
                      toast.info('Template download would start here');
                    }}>
                      Download Template
                    </Button>
                  </CardFooter>
                </Card>
              </div>

              <div className="md:col-span-1">
                <Card>
                  <CardHeader>
                    <CardTitle>Import Guide</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-start space-x-2">
                        <div className="bg-blue-100 rounded-full p-1">
                          <span className="font-bold text-blue-700">1</span>
                        </div>
                        <div className="text-sm">
                          <p className="font-medium">Prepare your data</p>
                          <p className="text-muted-foreground">
                            Ensure your CSV follows the format specified, with questions and answers properly formatted.
                          </p>
                        </div>
                      </div>

                      <div className="flex items-start space-x-2">
                        <div className="bg-blue-100 rounded-full p-1">
                          <span className="font-bold text-blue-700">2</span>
                        </div>
                        <div className="text-sm">
                          <p className="font-medium">Upload CSV files</p>
                          <p className="text-muted-foreground">
                            Drag & drop or select your CSV files to upload them to the system.
                          </p>
                        </div>
                      </div>

                      <div className="flex items-start space-x-2">
                        <div className="bg-blue-100 rounded-full p-1">
                          <span className="font-bold text-blue-700">3</span>
                        </div>
                        <div className="text-sm">
                          <p className="font-medium">Process the uploaded data</p>
                          <p className="text-muted-foreground">
                            After uploading, click "Process Imported Data" to start the embedding and clustering pipeline.
                          </p>
                        </div>
                      </div>

                      <div className="flex items-start space-x-2">
                        <div className="bg-blue-100 rounded-full p-1">
                          <span className="font-bold text-blue-700">4</span>
                        </div>
                        <div className="text-sm">
                          <p className="font-medium">Check results</p>
                          <p className="text-muted-foreground">
                            Once processing is complete, explore the results in the Dashboard and Cluster Explorer.
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Alert className="mt-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Processing Time</AlertTitle>
                  <AlertDescription>
                    Processing may take several minutes depending on the dataset size. You can check the status in the Import History tab.
                  </AlertDescription>
                </Alert>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="history">
            <Card>
              <CardHeader>
                <CardTitle>Import History</CardTitle>
                <CardDescription>
                  View the history of dataset imports and their processing status
                </CardDescription>
              </CardHeader>
              <CardContent>
                {historyLoading ? (
                  <div className="flex justify-center items-center h-40">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
                  </div>
                ) : importHistory.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>File Name</TableHead>
                        <TableHead>Upload Date</TableHead>
                        <TableHead>Records</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {importHistory.map((item) => (
                        <TableRow key={item.id}>
                          <TableCell className="font-medium">{item.filename}</TableCell>
                          <TableCell>{formatDate(item.uploadedAt)}</TableCell>
                          <TableCell>{item.recordCount}</TableCell>
                          <TableCell>{getStatusBadge(item.status)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No import history found. Upload a dataset to get started.
                  </div>
                )}
              </CardContent>
              <CardFooter>
                <Button
                  variant="outline"
                  size="sm"
                  className="ml-auto"
                  onClick={async () => {
                    setHistoryLoading(true);
                    try {
                      const history = await apiService.getImportHistory();
                      if (history) {
                        setImportHistory(history);
                      }
                      toast.success('Import history refreshed');
                    } catch (error) {
                      console.error('Error refreshing import history:', error);
                      toast.error('Failed to refresh import history');
                    } finally {
                      setHistoryLoading(false);
                    }
                  }}
                >
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Refresh History
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </MainLayout>
  );
}
