'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2 } from 'lucide-react';

export default function TestPage() {
    const [summaryData, setSummaryData] = useState<any>(null);
    const [productsData, setProductsData] = useState<any[]>([]);
    const [clustersData, setClustersData] = useState<any[]>([]);
    const [loading, setLoading] = useState<{ [key: string]: boolean }>({
        summary: false,
        products: false,
        clusters: false,
    });
    const [error, setError] = useState<{ [key: string]: string | null }>({
        summary: null,
        products: null,
        clusters: null,
    });

    // API base URL - adjust based on your FastAPI server address
    const API_BASE_URL = 'http://localhost:8000';

    // Function to fetch summary data
    const fetchSummary = async () => {
        setLoading((prev) => ({ ...prev, summary: true }));
        setError((prev) => ({ ...prev, summary: null }));

        try {
            const response = await fetch(`${API_BASE_URL}/summary`);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch summary data');
            }

            const data = await response.json();
            setSummaryData(data);
        } catch (err: any) {
            console.error('Error fetching summary:', err);
            setError((prev) => ({ ...prev, summary: err.message }));
        } finally {
            setLoading((prev) => ({ ...prev, summary: false }));
        }
    };

    // Function to fetch products data
    const fetchProducts = async () => {
        setLoading((prev) => ({ ...prev, products: true }));
        setError((prev) => ({ ...prev, products: null }));

        try {
            const response = await fetch(`${API_BASE_URL}/products`);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch products data');
            }

            const data = await response.json();
            setProductsData(data);
        } catch (err: any) {
            console.error('Error fetching products:', err);
            setError((prev) => ({ ...prev, products: err.message }));
        } finally {
            setLoading((prev) => ({ ...prev, products: false }));
        }
    };

    // Function to fetch clusters data
    const fetchClusters = async () => {
        setLoading((prev) => ({ ...prev, clusters: true }));
        setError((prev) => ({ ...prev, clusters: null }));

        try {
            const response = await fetch(`${API_BASE_URL}/clusters?limit=10`);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch clusters data');
            }

            const data = await response.json();
            setClustersData(data);
        } catch (err: any) {
            console.error('Error fetching clusters:', err);
            setError((prev) => ({ ...prev, clusters: err.message }));
        } finally {
            setLoading((prev) => ({ ...prev, clusters: false }));
        }
    };

    return (
        <div className="container py-8">
            <h1 className="text-3xl font-bold mb-6">FastAPI Integration Test</h1>

            {/* Summary Section */}
            <section className="mb-8">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-2xl font-semibold">Summary Data</h2>
                    <Button
                        onClick={fetchSummary}
                        disabled={loading.summary}
                    >
                        {loading.summary ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Loading...
                            </>
                        ) : 'Fetch Summary'}
                    </Button>
                </div>

                {error.summary && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                        <p>Error: {error.summary}</p>
                    </div>
                )}

                {summaryData ? (
                    <Card>
                        <CardHeader>
                            <CardTitle>Summary Information</CardTitle>
                            <CardDescription>Overall statistics from the API</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-96">
                                {JSON.stringify(summaryData, null, 2)}
                            </pre>
                        </CardContent>
                    </Card>
                ) : (
                    <div className="text-gray-500">Click the button to fetch summary data</div>
                )}
            </section>

            {/* Products Section */}
            <section className="mb-8">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-2xl font-semibold">Products Data</h2>
                    <Button
                        onClick={fetchProducts}
                        disabled={loading.products}
                    >
                        {loading.products ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Loading...
                            </>
                        ) : 'Fetch Products'}
                    </Button>
                </div>

                {error.products && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                        <p>Error: {error.products}</p>
                    </div>
                )}

                {productsData.length > 0 ? (
                    <Card>
                        <CardHeader>
                            <CardTitle>Products Information</CardTitle>
                            <CardDescription>List of products from the API</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-96">
                                {JSON.stringify(productsData, null, 2)}
                            </pre>
                        </CardContent>
                    </Card>
                ) : (
                    <div className="text-gray-500">Click the button to fetch products data</div>
                )}
            </section>

            {/* Clusters Section */}
            <section className="mb-8">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-2xl font-semibold">Clusters Data</h2>
                    <Button
                        onClick={fetchClusters}
                        disabled={loading.clusters}
                    >
                        {loading.clusters ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Loading...
                            </>
                        ) : 'Fetch Clusters'}
                    </Button>
                </div>

                {error.clusters && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                        <p>Error: {error.clusters}</p>
                    </div>
                )}

                {clustersData.length > 0 ? (
                    <Card>
                        <CardHeader>
                            <CardTitle>Clusters Information</CardTitle>
                            <CardDescription>First 10 clusters from the API</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-96">
                                {JSON.stringify(clustersData, null, 2)}
                            </pre>
                        </CardContent>
                    </Card>
                ) : (
                    <div className="text-gray-500">Click the button to fetch clusters data</div>
                )}
            </section>
        </div>
    );
}