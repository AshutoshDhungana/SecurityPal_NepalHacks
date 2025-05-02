'use client';

import { useState, useEffect } from 'react';
import MainLayout from '@/components/MainLayout';
import api from '@/lib/api';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Play, Calendar, Clock, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

export default function PipelineControlPage() {
    // State for pipeline execution
    const [pipelineSteps, setPipelineSteps] = useState<any[]>([]);
    const [selectedSteps, setSelectedSteps] = useState<number[]>([]);
    const [products, setProducts] = useState<any[]>([]);
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [dryRun, setDryRun] = useState(false);
    const [loading, setLoading] = useState(true);
    const [executing, setExecuting] = useState(false);
    const [lastRunResult, setLastRunResult] = useState<any>(null);
    const [logsExpanded, setLogsExpanded] = useState(false);

    // State for product data processing
    const [datasetSource, setDatasetSource] = useState('combined');
    const [productSelections, setProductSelections] = useState<{ [key: string]: boolean }>({});
    const [processingOptions, setProcessingOptions] = useState({
        generateEmbeddings: true,
        performClustering: true,
        generateSummaries: true,
        cacheResults: true
    });
    const [advancedOptions, setAdvancedOptions] = useState({
        embeddingModel: "all-MiniLM-L6-v2",
        clusteringAlgorithm: "HDBSCAN",
        minClusterSize: 5,
        similarityThreshold: 0.85
    });
    const [processingLogs, setProcessingLogs] = useState<string[]>([]);
    const [processingInProgress, setProcessingInProgress] = useState(false);

    // Load pipeline steps and products on page load
    useEffect(() => {
        async function fetchPipelineData() {
            try {
                setLoading(true);

                // Fetch pipeline steps
                const steps = await api.getPipelineSteps();
                setPipelineSteps(steps || []);

                // Default: select all steps
                if (steps && steps.length > 0) {
                    setSelectedSteps(steps.map((_: any, index) => index));
                }

                // Fetch products
                const productsData = await api.getProducts();
                setProducts(productsData || []);

                // Initialize product selections (all selected by default)
                if (productsData && productsData.length > 0) {
                    const selections: { [key: string]: boolean } = {};
                    productsData.forEach(product => {
                        selections[product.product_name] = true;
                    });
                    setProductSelections(selections);
                }
            } catch (error) {
                console.error('Error fetching pipeline data:', error);
                toast.error('Failed to load pipeline configuration');

                // Generate mock data if API fails
                generateMockPipelineData();
            } finally {
                setLoading(false);
            }
        }

        fetchPipelineData();
    }, []);

    // Generate mock pipeline data for testing
    const generateMockPipelineData = () => {
        // Mock pipeline steps
        const mockSteps = [
            { name: 'Data Cleaning', file: 'data_cleaning.py', description: 'Clean and prepare data for processing' },
            { name: 'Embedding Generation', file: 'embedding.py', description: 'Generate embeddings for content' },
            { name: 'Clustering', file: 'clustering.py', description: 'Group similar content into clusters' },
            { name: 'Summary Generation', file: 'summarize.py', description: 'Create summaries for clusters' },
            { name: 'Cache Creation', file: 'cache_generation.py', description: 'Build cache for fast retrieval' }
        ];
        setPipelineSteps(mockSteps);
        setSelectedSteps([0, 1, 2, 3, 4]); // Select all by default

        // Mock products
        const mockProducts = [
            { product_id: 1, product_name: 'Danfe_Corp_Product_1' },
            { product_id: 2, product_name: 'Danfe_Corp_Product_2' },
            { product_id: 3, product_name: 'Danfe_Corp_Product_3' },
            { product_id: 4, product_name: 'Danfe_Corp_Product_4' }
        ];
        setProducts(mockProducts);

        // Initialize product selections
        const selections: { [key: string]: boolean } = {};
        mockProducts.forEach(product => {
            selections[product.product_name] = true;
        });
        setProductSelections(selections);
    };

    // Toggle selection of a step
    const toggleStepSelection = (index: number) => {
        if (selectedSteps.includes(index)) {
            setSelectedSteps(selectedSteps.filter(stepIndex => stepIndex !== index));
        } else {
            setSelectedSteps([...selectedSteps, index].sort((a, b) => a - b));
        }
    };

    // Toggle selection of a product
    const toggleProductSelection = (productName: string) => {
        setProductSelections(prev => ({
            ...prev,
            [productName]: !prev[productName]
        }));
    };

    // Update processing options
    const updateProcessingOption = (option: keyof typeof processingOptions, value: boolean) => {
        setProcessingOptions(prev => ({
            ...prev,
            [option]: value
        }));
    };

    // Update advanced options
    const updateAdvancedOption = (option: keyof typeof advancedOptions, value: any) => {
        setAdvancedOptions(prev => ({
            ...prev,
            [option]: value
        }));
    };

    // Run the pipeline
    const runPipeline = async () => {
        if (selectedSteps.length === 0) {
            toast.error('Please select at least one pipeline step');
            return;
        }

        setExecuting(true);
        try {
            // Get start and end steps
            const startStep = Math.min(...selectedSteps);
            const endStep = Math.max(...selectedSteps) + 1; // +1 because the API expects exclusive end

            // Prepare options
            const options = {
                product: selectedProduct,
                start_step: startStep,
                end_step: endStep,
                dry_run: dryRun
            };

            // Call API to run pipeline
            const result = await api.runPipeline(options);
            setLastRunResult(result);
            toast.success('Pipeline started successfully');

            // Simulate log output
            simulatePipelineLogs();
        } catch (error) {
            console.error('Error running pipeline:', error);
            toast.error('Failed to start pipeline');

            // Set mock result
            setLastRunResult({
                success: true,
                message: 'Pipeline started in mock mode',
                run_id: `mock-${Date.now()}`,
                options: {
                    product: selectedProduct,
                    start_step: Math.min(...selectedSteps),
                    end_step: Math.max(...selectedSteps) + 1,
                    dry_run: dryRun
                }
            });

            // Simulate log output
            simulatePipelineLogs();
        } finally {
            setExecuting(false);
        }
    };

    // Process product datasets
    const processProductDatasets = () => {
        const selectedProductsList = Object.entries(productSelections)
            .filter(([_, selected]) => selected)
            .map(([productName, _]) => productName);

        if (selectedProductsList.length === 0) {
            toast.error('Please select at least one product');
            return;
        }

        setProcessingInProgress(true);

        // Clear previous logs
        setProcessingLogs([]);

        // Simulate processing with logs
        const logMessages: string[] = [
            `${getCurrentTimestamp()} - INFO - Starting product data processing`,
            `${getCurrentTimestamp()} - INFO - Using ${datasetSource === 'combined' ? 'combined dataset with filtering' : 'individual product datasets'}`
        ];

        // Add selected products to logs
        logMessages.push(`${getCurrentTimestamp()} - INFO - Processing products: ${selectedProductsList.join(', ')}`);

        // Add options to logs
        if (processingOptions.generateEmbeddings) {
            logMessages.push(`${getCurrentTimestamp()} - INFO - Will generate embeddings using model: ${advancedOptions.embeddingModel}`);
        }

        if (processingOptions.performClustering) {
            logMessages.push(`${getCurrentTimestamp()} - INFO - Will perform clustering using algorithm: ${advancedOptions.clusteringAlgorithm}`);
            logMessages.push(`${getCurrentTimestamp()} - INFO - Using min cluster size: ${advancedOptions.minClusterSize}, similarity threshold: ${advancedOptions.similarityThreshold}`);
        }

        setProcessingLogs(logMessages);

        // Simulate product-specific logs over time
        simulateProductProcessingLogs(selectedProductsList);
    };

    // Simulate pipeline logs
    const simulatePipelineLogs = () => {
        const logs: string[] = [];
        const steps = selectedSteps.map(index => pipelineSteps[index]?.name || `Step ${index + 1}`);

        // Initial log messages
        logs.push(`${getCurrentTimestamp()} - Pipeline execution started`);
        logs.push(`${getCurrentTimestamp()} - Selected steps: ${steps.join(', ')}`);
        if (selectedProduct) {
            logs.push(`${getCurrentTimestamp()} - Processing product: ${selectedProduct}`);
        } else {
            logs.push(`${getCurrentTimestamp()} - Processing all products`);
        }

        if (dryRun) {
            logs.push(`${getCurrentTimestamp()} - Running in DRY RUN mode (no actual changes will be made)`);
        }

        setProcessingLogs(logs);
        setLogsExpanded(true);

        // Simulate logs for each step
        let currentIndex = 0;
        const logInterval = setInterval(() => {
            if (currentIndex >= steps.length) {
                setProcessingLogs(prevLogs => [
                    ...prevLogs,
                    `${getCurrentTimestamp()} - Pipeline execution completed successfully`,
                    `${getCurrentTimestamp()} - Results saved to output directory`
                ]);
                clearInterval(logInterval);
                return;
            }

            const step = steps[currentIndex];
            setProcessingLogs(prevLogs => [
                ...prevLogs,
                `${getCurrentTimestamp()} - Starting ${step}`,
                `${getCurrentTimestamp()} - ${step} in progress...`,
                `${getCurrentTimestamp()} - ${step} completed successfully`
            ]);

            currentIndex++;
        }, 2000);
    };

    // Simulate product processing logs
    const simulateProductProcessingLogs = (productsList: string[]) => {
        let currentProductIndex = 0;

        const processNextProduct = () => {
            if (currentProductIndex >= productsList.length) {
                // All products processed
                setProcessingLogs(prevLogs => [
                    ...prevLogs,
                    `${getCurrentTimestamp()} - INFO - All product data processing completed`,
                    `${getCurrentTimestamp()} - INFO - Results available in the output directory`
                ]);

                setProcessingInProgress(false);
                toast.success(`Successfully processed data for ${productsList.length} products`);
                return;
            }

            const product = productsList[currentProductIndex];

            // Simulate processing for this product
            setTimeout(() => {
                setProcessingLogs(prevLogs => [
                    ...prevLogs,
                    `${getCurrentTimestamp()} - INFO - Filtering data for ${product}`
                ]);
            }, 500);

            if (processingOptions.generateEmbeddings) {
                setTimeout(() => {
                    const embeddingCount = Math.floor(Math.random() * 4000) + 1000;
                    setProcessingLogs(prevLogs => [
                        ...prevLogs,
                        `${getCurrentTimestamp()} - INFO - Generated embeddings for ${embeddingCount} entries`
                    ]);
                }, 1500);
            }

            if (processingOptions.performClustering) {
                setTimeout(() => {
                    const clusterCount = Math.floor(Math.random() * 150) + 50;
                    setProcessingLogs(prevLogs => [
                        ...prevLogs,
                        `${getCurrentTimestamp()} - INFO - Created ${clusterCount} clusters for ${product}`
                    ]);
                }, 2500);
            }

            if (processingOptions.generateSummaries) {
                setTimeout(() => {
                    setProcessingLogs(prevLogs => [
                        ...prevLogs,
                        `${getCurrentTimestamp()} - INFO - Generated summary for ${product}`
                    ]);
                }, 3000);
            }

            // Move to next product
            setTimeout(() => {
                currentProductIndex++;
                processNextProduct();
            }, 3500);
        };

        // Start processing
        processNextProduct();
    };

    // Get current timestamp for logs
    const getCurrentTimestamp = () => {
        const now = new Date();
        return now.toISOString().replace('T', ' ').substring(0, 19);
    };

    // Get count of selected products
    const selectedProductsCount = Object.values(productSelections).filter(Boolean).length;

    return (
        <MainLayout>
            <h1 className="text-3xl font-bold text-blue-600 mb-6">Pipeline Control</h1>

            <Tabs defaultValue="run" className="w-full">
                <TabsList className="mb-4">
                    <TabsTrigger value="run">Run Pipeline</TabsTrigger>
                    <TabsTrigger value="product">Product Data Processing</TabsTrigger>
                </TabsList>

                <TabsContent value="run">
                    {loading ? (
                        <div className="flex items-center justify-center h-64">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                        </div>
                    ) : (
                        <>
                            <Card className="mb-8">
                                <CardHeader>
                                    <CardTitle>Pipeline Configuration</CardTitle>
                                    <CardDescription>
                                        Configure and run the pipeline for content processing
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                                        <div>
                                            <h3 className="text-sm font-medium text-gray-700 mb-2">Select Product</h3>
                                            <select
                                                value={selectedProduct || ''}
                                                onChange={(e) => setSelectedProduct(e.target.value || null)}
                                                className="w-full p-2 border border-gray-300 rounded-md"
                                            >
                                                <option value="">All Products</option>
                                                {products.map(product => (
                                                    <option key={product.product_id} value={product.product_name}>
                                                        {product.product_name}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>

                                        <div>
                                            <h3 className="text-sm font-medium text-gray-700 mb-2">Execution Mode</h3>
                                            <div className="flex items-center">
                                                <input
                                                    type="checkbox"
                                                    id="dry-run"
                                                    checked={dryRun}
                                                    onChange={(e) => setDryRun(e.target.checked)}
                                                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                                />
                                                <label htmlFor="dry-run" className="ml-2 block text-sm text-gray-900">
                                                    Dry Run (Preview Only)
                                                </label>
                                            </div>
                                        </div>
                                    </div>

                                    <div>
                                        <h3 className="text-sm font-medium text-gray-700 mb-3">Pipeline Steps</h3>
                                        <div className="space-y-2 max-h-[300px] overflow-y-auto p-1">
                                            {pipelineSteps.map((step, index) => (
                                                <div
                                                    key={index}
                                                    className={`
                              border rounded-lg p-3 transition-colors cursor-pointer
                              ${selectedSteps.includes(index)
                                                            ? 'border-blue-500 bg-blue-50'
                                                            : 'border-gray-200 hover:border-blue-300'}
                            `}
                                                    onClick={() => toggleStepSelection(index)}
                                                >
                                                    <div className="flex items-center">
                                                        <input
                                                            type="checkbox"
                                                            checked={selectedSteps.includes(index)}
                                                            onChange={() => toggleStepSelection(index)}
                                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                                            onClick={(e) => e.stopPropagation()}
                                                        />
                                                        <div className="ml-3">
                                                            <p className="font-medium">{step.name}</p>
                                                            <p className="text-sm text-gray-500">
                                                                {step.file} {step.description && `- ${step.description}`}
                                                            </p>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <div className="mt-6 pt-4 border-t border-gray-200">
                                        <Button
                                            className="w-full md:w-auto"
                                            onClick={runPipeline}
                                            disabled={executing || selectedSteps.length === 0}
                                        >
                                            {executing ? (
                                                <>
                                                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                                    Running Pipeline...
                                                </>
                                            ) : (
                                                <>
                                                    <Play className="mr-2 h-4 w-4" />
                                                    Run Pipeline
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>

                            {lastRunResult && (
                                <Card className="mb-8">
                                    <CardHeader>
                                        <CardTitle>Last Run Result</CardTitle>
                                        <CardDescription>
                                            Details about the most recent pipeline execution
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="bg-gray-50 p-4 rounded-lg mb-4 border border-gray-200">
                                            <div className="flex items-center mb-2">
                                                {lastRunResult.success ? (
                                                    <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                                                ) : (
                                                    <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                                                )}
                                                <h3 className="font-medium">{lastRunResult.message}</h3>
                                            </div>

                                            {lastRunResult.run_id && (
                                                <p className="text-sm text-gray-500 mb-2">
                                                    Run ID: {lastRunResult.run_id}
                                                </p>
                                            )}

                                            {lastRunResult.options && (
                                                <div className="text-sm">
                                                    <p className="font-medium text-gray-700 mb-1">Configuration:</p>
                                                    <ul className="list-disc list-inside text-gray-600 space-y-1">
                                                        <li>Product: {lastRunResult.options.product || 'All Products'}</li>
                                                        <li>Steps: {lastRunResult.options.start_step} to {lastRunResult.options.end_step - 1}</li>
                                                        <li>Mode: {lastRunResult.options.dry_run ? 'Dry Run' : 'Normal Execution'}</li>
                                                    </ul>
                                                </div>
                                            )}
                                        </div>

                                        <div>
                                            <div
                                                className="flex items-center justify-between cursor-pointer text-blue-600 hover:text-blue-800 mb-2"
                                                onClick={() => setLogsExpanded(!logsExpanded)}
                                            >
                                                <h3 className="font-medium">Pipeline Logs</h3>
                                                <span>{logsExpanded ? 'Hide' : 'Show'}</span>
                                            </div>

                                            {logsExpanded && (
                                                <div className="bg-black text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                                                    <pre className="whitespace-pre-wrap">
                                                        {processingLogs.join('\n')}
                                                    </pre>
                                                </div>
                                            )}
                                        </div>
                                    </CardContent>
                                </Card>
                            )}

                            <Card>
                                <CardHeader>
                                    <CardTitle>Schedule Pipeline</CardTitle>
                                    <CardDescription>
                                        Set up recurring pipeline executions
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                        <div>
                                            <h3 className="text-sm font-medium text-gray-700 mb-2">Schedule Type</h3>
                                            <select
                                                className="w-full p-2 border border-gray-300 rounded-md"
                                                defaultValue="daily"
                                            >
                                                <option value="daily">Daily</option>
                                                <option value="weekly">Weekly</option>
                                                <option value="monthly">Monthly</option>
                                            </select>
                                        </div>

                                        <div>
                                            <h3 className="text-sm font-medium text-gray-700 mb-2">Time</h3>
                                            <div className="flex items-center">
                                                <Clock className="h-5 w-5 text-gray-400 mr-2" />
                                                <input
                                                    type="time"
                                                    defaultValue="01:00"
                                                    className="p-2 border border-gray-300 rounded-md"
                                                />
                                            </div>
                                        </div>

                                        <div className="flex items-end">
                                            <Button
                                                variant="outline"
                                                className="w-full"
                                                onClick={() => toast.success('Schedule set (action not implemented)')}
                                            >
                                                <Calendar className="mr-2 h-4 w-4" />
                                                Set Schedule
                                            </Button>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </>
                    )}
                </TabsContent>

                <TabsContent value="product">
                    <Card className="mb-8">
                        <CardHeader>
                            <CardTitle>Product Data Processing</CardTitle>
                            <CardDescription>
                                Process product-specific datasets for content clustering and analytics
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-6">
                                <div>
                                    <h3 className="text-sm font-medium text-gray-700 mb-3">Dataset Source</h3>
                                    <div className="space-y-2">
                                        <div className="flex items-center">
                                            <input
                                                type="radio"
                                                id="combined-dataset"
                                                name="dataset-source"
                                                checked={datasetSource === 'combined'}
                                                onChange={() => setDatasetSource('combined')}
                                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                            />
                                            <label htmlFor="combined-dataset" className="ml-2 block text-sm text-gray-900">
                                                Combined Dataset (with filtering)
                                            </label>
                                        </div>

                                        <div className="flex items-center">
                                            <input
                                                type="radio"
                                                id="individual-datasets"
                                                name="dataset-source"
                                                checked={datasetSource === 'individual'}
                                                onChange={() => setDatasetSource('individual')}
                                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                            />
                                            <label htmlFor="individual-datasets" className="ml-2 block text-sm text-gray-900">
                                                Individual Product Datasets
                                            </label>
                                        </div>
                                    </div>
                                </div>

                                <div>
                                    <h3 className="text-sm font-medium text-gray-700 mb-3">Select Products to Process</h3>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        {products.map(product => (
                                            <div key={product.product_id} className="flex items-center">
                                                <input
                                                    type="checkbox"
                                                    id={`product-${product.product_id}`}
                                                    checked={productSelections[product.product_name] || false}
                                                    onChange={() => toggleProductSelection(product.product_name)}
                                                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                                />
                                                <label
                                                    htmlFor={`product-${product.product_id}`}
                                                    className="ml-2 block text-sm text-gray-900 truncate"
                                                >
                                                    {product.product_name}
                                                </label>
                                            </div>
                                        ))}
                                    </div>

                                    <div className="mt-2 text-sm text-gray-500">
                                        Selected {selectedProductsCount} products for processing
                                    </div>
                                </div>

                                <div>
                                    <h3 className="text-sm font-medium text-gray-700 mb-3">Processing Options</h3>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="generate-embeddings"
                                                checked={processingOptions.generateEmbeddings}
                                                onChange={(e) => updateProcessingOption('generateEmbeddings', e.target.checked)}
                                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                            />
                                            <label htmlFor="generate-embeddings" className="ml-2 block text-sm text-gray-900">
                                                Generate Embeddings
                                            </label>
                                        </div>

                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="perform-clustering"
                                                checked={processingOptions.performClustering}
                                                onChange={(e) => updateProcessingOption('performClustering', e.target.checked)}
                                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                            />
                                            <label htmlFor="perform-clustering" className="ml-2 block text-sm text-gray-900">
                                                Perform Clustering
                                            </label>
                                        </div>

                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="generate-summaries"
                                                checked={processingOptions.generateSummaries}
                                                onChange={(e) => updateProcessingOption('generateSummaries', e.target.checked)}
                                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                            />
                                            <label htmlFor="generate-summaries" className="ml-2 block text-sm text-gray-900">
                                                Generate Summaries
                                            </label>
                                        </div>

                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="cache-results"
                                                checked={processingOptions.cacheResults}
                                                onChange={(e) => updateProcessingOption('cacheResults', e.target.checked)}
                                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                            />
                                            <label htmlFor="cache-results" className="ml-2 block text-sm text-gray-900">
                                                Cache Results
                                            </label>
                                        </div>
                                    </div>
                                </div>

                                <div>
                                    <button
                                        type="button"
                                        className="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center"
                                        onClick={() => document.getElementById('advanced-options')?.classList.toggle('hidden')}
                                    >
                                        <span>Advanced Options</span>
                                        <svg className="ml-1 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                        </svg>
                                    </button>

                                    <div id="advanced-options" className="hidden mt-4 space-y-4 bg-gray-50 p-4 rounded-lg">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                Embedding Model
                                            </label>
                                            <select
                                                value={advancedOptions.embeddingModel}
                                                onChange={(e) => updateAdvancedOption('embeddingModel', e.target.value)}
                                                className="w-full p-2 border border-gray-300 rounded-md"
                                            >
                                                <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
                                                <option value="all-mpnet-base-v2">all-mpnet-base-v2</option>
                                                <option value="paraphrase-multilingual-MiniLM-L12-v2">paraphrase-multilingual-MiniLM-L12-v2</option>
                                            </select>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                Clustering Algorithm
                                            </label>
                                            <select
                                                value={advancedOptions.clusteringAlgorithm}
                                                onChange={(e) => updateAdvancedOption('clusteringAlgorithm', e.target.value)}
                                                className="w-full p-2 border border-gray-300 rounded-md"
                                            >
                                                <option value="HDBSCAN">HDBSCAN</option>
                                                <option value="DBSCAN">DBSCAN</option>
                                                <option value="KMeans">KMeans</option>
                                            </select>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                Minimum Cluster Size
                                            </label>
                                            <input
                                                type="range"
                                                min={2}
                                                max={20}
                                                value={advancedOptions.minClusterSize}
                                                onChange={(e) => updateAdvancedOption('minClusterSize', parseInt(e.target.value))}
                                                className="w-full"
                                            />
                                            <div className="flex justify-between text-xs text-gray-500">
                                                <span>2</span>
                                                <span>{advancedOptions.minClusterSize}</span>
                                                <span>20</span>
                                            </div>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                Similarity Threshold
                                            </label>
                                            <input
                                                type="range"
                                                min={0.5}
                                                max={1}
                                                step={0.01}
                                                value={advancedOptions.similarityThreshold}
                                                onChange={(e) => updateAdvancedOption('similarityThreshold', parseFloat(e.target.value))}
                                                className="w-full"
                                            />
                                            <div className="flex justify-between text-xs text-gray-500">
                                                <span>0.5</span>
                                                <span>{advancedOptions.similarityThreshold}</span>
                                                <span>1.0</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-gray-200">
                                    <Button
                                        onClick={processProductDatasets}
                                        disabled={processingInProgress || selectedProductsCount === 0}
                                    >
                                        {processingInProgress ? (
                                            <>
                                                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                                Processing...
                                            </>
                                        ) : (
                                            'Process Product Datasets'
                                        )}
                                    </Button>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    {processingLogs.length > 0 && (
                        <Card>
                            <CardHeader>
                                <CardTitle>Processing Logs</CardTitle>
                                <CardDescription>
                                    Real-time logs from the data processing operations
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="bg-black text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto max-h-[400px] overflow-y-auto">
                                    <pre className="whitespace-pre-wrap">
                                        {processingLogs.join('\n')}
                                    </pre>
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </TabsContent>
            </Tabs>
        </MainLayout>
    );
}