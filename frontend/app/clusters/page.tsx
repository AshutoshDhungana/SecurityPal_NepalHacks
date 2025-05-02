'use client';

import { useState, useEffect } from 'react';
import MainLayout from '@/components/MainLayout';
import api from '@/lib/api';
import { ChevronDown, ChevronRight, RefreshCw, Search, Filter, Database } from 'lucide-react';
import { toast } from 'sonner';

export default function ClustersPage() {
    const [clusters, setClusters] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [minSize, setMinSize] = useState(5);
    const [healthStatus, setHealthStatus] = useState('All');
    const [limit, setLimit] = useState(50);
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
    const [clusterEntries, setClusterEntries] = useState<any[]>([]);
    const [loadingEntries, setLoadingEntries] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [products, setProducts] = useState<any[]>([]);
    const [showFilters, setShowFilters] = useState(true);

    // Load products for filter
    useEffect(() => {
        async function fetchProducts() {
            try {
                const data = await api.getProducts();
                setProducts(data || []);
            } catch (error) {
                console.error('Error fetching products:', error);
            }
        }
        fetchProducts();
    }, []);

    // Load clusters based on filters
    useEffect(() => {
        async function fetchClusters() {
            try {
                setLoading(true);

                // Prepare parameters
                const params: any = {
                    limit,
                    offset: 0,
                    min_size: minSize
                };

                if (selectedProduct) {
                    // Format product name with underscores instead of spaces
                    params.product = selectedProduct.replace(' ', '_');
                }

                if (healthStatus !== 'All') {
                    params.health_status = healthStatus;
                }

                // Fetch clusters
                const data = await api.getClusters(params);
                setClusters(data || []);
            } catch (error: any) {
                console.error('Error fetching clusters:', error);
                toast.error('Failed to load clusters');
            } finally {
                setLoading(false);
            }
        }

        fetchClusters();
    }, [selectedProduct, minSize, healthStatus, limit]);

    // Load cluster entries when a cluster is selected
    useEffect(() => {
        async function fetchClusterEntries() {
            if (!selectedClusterId) return;

            try {
                setLoadingEntries(true);
                const data = await api.getClusterEntries(selectedClusterId);
                setClusterEntries(data || []);
            } catch (error: any) {
                console.error(`Error fetching entries for cluster ${selectedClusterId}:`, error);
                toast.error('Failed to load cluster entries');
                // Use mock data if API fails
                setClusterEntries(getMockEntries(selectedClusterId, 7));
            } finally {
                setLoadingEntries(false);
            }
        }

        fetchClusterEntries();
    }, [selectedClusterId]);

    // Generate mock entries for testing when API fails
    function getMockEntries(clusterId: string, size: number = 5) {
        const mockEntries = [];

        // Add a canonical question
        mockEntries.push({
            question: `What is Cluster ${clusterId}?`,
            answer: `This is the canonical answer about Cluster ${clusterId}. It provides the most accurate and comprehensive information about this topic.`,
            is_canonical: true,
            last_updated: new Date().toISOString().split('T')[0],
        });

        // Add some regular questions
        for (let i = 0; i < size - 1; i++) {
            mockEntries.push({
                question: `Question ${i + 1} about Cluster ${clusterId}?`,
                answer: `This is answer ${i + 1} about Cluster ${clusterId}. This provides additional information on specific aspects of this topic.`,
                is_canonical: false,
                last_updated: new Date(Date.now() - i * 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            });
        }

        return mockEntries;
    }

    // Filter cluster entries by search query
    const filteredEntries = searchQuery
        ? clusterEntries.filter(entry =>
            entry.question?.toLowerCase().includes(searchQuery.toLowerCase()))
        : clusterEntries;

    // Count canonical and regular entries
    const canonicalCount = clusterEntries.filter(e => e.is_canonical).length;
    const regularCount = clusterEntries.length - canonicalCount;

    // Get cluster name for display
    const getClusterName = (cluster: any, index: number) => {
        if (cluster.canonical_questions && cluster.canonical_questions.length) {
            const question = cluster.canonical_questions[0];
            return question.length > 70 ? `${question.substring(0, 70)}...` : question;
        } else if (cluster.questions && cluster.questions.length) {
            const question = cluster.questions[0];
            return question.length > 70 ? `${question.substring(0, 70)}...` : question;
        }
        return `Cluster ${index + 1}`;
    };

    return (
        <MainLayout>
            <div className="container mx-auto">
                <h1 className="text-3xl font-bold text-blue-600 mb-6">Cluster Explorer</h1>

                {/* Filters */}
                <div className="bg-white rounded-lg shadow-md p-6 mb-8">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-semibold text-gray-800">Filter Clusters</h2>
                        <button
                            className="text-sm text-blue-600 hover:underline flex items-center"
                            onClick={() => setShowFilters(!showFilters)}
                        >
                            <Filter size={16} className="mr-1" /> {showFilters ? 'Hide' : 'Show'} Filters
                        </button>
                    </div>
                    {showFilters && (
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Min Cluster Size
                                </label>
                                <input
                                    type="number"
                                    min={2}
                                    value={minSize}
                                    onChange={(e) => setMinSize(parseInt(e.target.value) || 2)}
                                    className="w-full p-2 border border-gray-300 rounded-md"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Health Status
                                </label>
                                <select
                                    value={healthStatus}
                                    onChange={(e) => setHealthStatus(e.target.value)}
                                    className="w-full p-2 border border-gray-300 rounded-md"
                                >
                                    <option value="All">All</option>
                                    <option value="Healthy">Healthy</option>
                                    <option value="Needs Review">Needs Review</option>
                                    <option value="Critical">Critical</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Product
                                </label>
                                <select
                                    value={selectedProduct || ''}
                                    onChange={(e) => setSelectedProduct(e.target.value || null)}
                                    className="w-full p-2 border border-gray-300 rounded-md"
                                >
                                    <option value="">All Products</option>
                                    {products.map((product: any) => (
                                        <option key={product.id} value={product.name}>
                                            {product.name}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Max Clusters to Show
                                </label>
                                <input
                                    type="range"
                                    min={10}
                                    max={100}
                                    step={10}
                                    value={limit}
                                    onChange={(e) => setLimit(parseInt(e.target.value))}
                                    className="w-full"
                                />
                                <div className="text-center text-sm text-gray-500">{limit} clusters</div>
                            </div>
                        </div>
                    )}
                </div>

                {loading ? (
                    <div className="flex items-center justify-center h-64">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                    </div>
                ) : (
                    <div className="flex flex-col lg:flex-row gap-8">
                        {/* Clusters List */}
                        <div className="w-full lg:w-1/2">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4">
                                {clusters.length > 0 ? `Found ${clusters.length} clusters` : 'No clusters found'}
                            </h2>

                            <div className="space-y-4 max-h-[700px] overflow-y-auto pr-4">
                                {clusters.map((cluster, index) => {
                                    // Only include valid clusters (exclude cluster_id = -1)
                                    if (cluster.cluster_id === -1) return null;

                                    const clusterId = cluster.cluster_id;
                                    const clusterName = getClusterName(cluster, index);
                                    const health = cluster.health_status || 'Unknown';
                                    const isSelected = selectedClusterId === clusterId;

                                    // Determine health status badge color
                                    let healthBadgeColor = 'bg-gray-200 text-gray-800';
                                    if (health === 'Healthy') healthBadgeColor = 'bg-green-500 text-white';
                                    else if (health === 'Needs Review') healthBadgeColor = 'bg-orange-500 text-white';
                                    else if (health === 'Critical') healthBadgeColor = 'bg-red-500 text-white';

                                    return (
                                        <div
                                            key={clusterId}
                                            className={`
                        border rounded-lg overflow-hidden transition-all
                        ${isSelected ? 'border-blue-500 shadow-md' : 'border-gray-200 hover:border-blue-300'}
                      `}
                                        >
                                            <div
                                                className="p-4 cursor-pointer flex items-start gap-3"
                                                onClick={() => setSelectedClusterId(isSelected ? null : clusterId)}
                                            >
                                                <div className="mt-1">
                                                    {isSelected ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                                                </div>
                                                <div className="flex-1">
                                                    <h3 className="font-medium text-gray-900">{clusterName}</h3>
                                                    <div className="text-sm text-gray-500 mt-1">
                                                        ID: {clusterId} | Size: {cluster.size || 0} entries |
                                                        <span className={`ml-1 px-2 py-0.5 rounded-full text-xs ${healthBadgeColor}`}>
                                                            {health}
                                                        </span> |
                                                        Score: {(cluster.similarity_score || 0).toFixed(2)}
                                                    </div>
                                                </div>
                                            </div>

                                            {isSelected && (
                                                <div className="bg-gray-50 p-4 border-t border-gray-200">
                                                    <div className="flex items-center justify-between mb-3">
                                                        <h4 className="font-medium">Cluster Details</h4>
                                                        <button
                                                            className="text-sm text-blue-600 hover:underline flex items-center"
                                                            onClick={() => {
                                                                setSelectedClusterId(null);
                                                                setTimeout(() => setSelectedClusterId(clusterId), 100);
                                                            }}
                                                        >
                                                            <RefreshCw size={14} className="mr-1" /> Refresh
                                                        </button>
                                                    </div>

                                                    <ul className="text-sm space-y-1 mb-3">
                                                        <li><span className="font-medium">ID:</span> {clusterId}</li>
                                                        <li><span className="font-medium">Size:</span> {cluster.size || 0} entries</li>
                                                        <li><span className="font-medium">Health:</span> {health}</li>
                                                        <li><span className="font-medium">Similarity Score:</span> {(cluster.similarity_score || 0).toFixed(2)}</li>
                                                        <li><span className="font-medium">Product:</span> {cluster.product || 'Unknown'}</li>
                                                    </ul>

                                                    {cluster.topics && cluster.topics.length > 0 && (
                                                        <>
                                                            <h5 className="font-medium text-sm mb-1">Key Topics</h5>
                                                            <ul className="list-disc list-inside text-sm">
                                                                {cluster.topics.map((topic: string, i: number) => (
                                                                    <li key={i}>{topic}</li>
                                                                ))}
                                                            </ul>
                                                        </>
                                                    )}

                                                    <button
                                                        className="mt-3 w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors text-sm flex items-center justify-center gap-2"
                                                        onClick={() => {
                                                            // Make sure the entries panel is visible on mobile
                                                            if (window.innerWidth < 1024) {
                                                                const element = document.getElementById('cluster-entries');
                                                                if (element) element.scrollIntoView({ behavior: 'smooth' });
                                                            }
                                                            // This ensures the entries load if they weren't loaded before
                                                            setSelectedClusterId(clusterId);
                                                        }}
                                                    >
                                                        <Database size={16} />
                                                        View All Questions
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>

                            {clusters.length === 0 && (
                                <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 p-4 rounded-lg">
                                    <p>No clusters found matching the criteria. Try adjusting the filters.</p>
                                </div>
                            )}
                        </div>

                        {/* Cluster Entries */}
                        <div id="cluster-entries" className="w-full lg:w-1/2">
                            {selectedClusterId ? (
                                loadingEntries ? (
                                    <div className="flex items-center justify-center h-64">
                                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                                        <span className="ml-3 text-blue-600">Loading questions...</span>
                                    </div>
                                ) : (
                                    <div className="bg-white rounded-lg shadow-md p-6">
                                        <h2 className="text-xl font-semibold text-gray-800 mb-4">
                                            Questions in Cluster {selectedClusterId}
                                        </h2>

                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                            <div className="bg-blue-50 rounded p-4 text-center">
                                                <div className="text-2xl font-bold text-blue-700">{clusterEntries.length}</div>
                                                <div className="text-sm text-blue-600">Total Questions</div>
                                            </div>

                                            <div className="bg-green-50 rounded p-4 text-center">
                                                <div className="text-2xl font-bold text-green-700">{canonicalCount}</div>
                                                <div className="text-sm text-green-600">Canonical</div>
                                            </div>

                                            <div className="bg-purple-50 rounded p-4 text-center">
                                                <div className="text-2xl font-bold text-purple-700">{regularCount}</div>
                                                <div className="text-sm text-purple-600">Regular</div>
                                            </div>
                                        </div>

                                        <div className="mb-4 relative">
                                            <input
                                                type="text"
                                                placeholder="Search in this cluster..."
                                                value={searchQuery}
                                                onChange={(e) => setSearchQuery(e.target.value)}
                                                className="w-full p-2 pl-10 border border-gray-300 rounded-md"
                                            />
                                            <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                                        </div>

                                        {filteredEntries.length > 0 ? (
                                            <div className="space-y-6">
                                                {/* Canonical Questions */}
                                                {filteredEntries.some(e => e.is_canonical) && (
                                                    <div>
                                                        <h3 className="font-medium text-gray-800 mb-3 flex items-center">
                                                            <span className="inline-block mr-2 w-2 h-2 rounded-full bg-green-500"></span>
                                                            Canonical Questions
                                                        </h3>
                                                        <div className="space-y-4">
                                                            {filteredEntries
                                                                .filter(entry => entry.is_canonical)
                                                                .map((entry, i) => (
                                                                    <div key={i} className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-600 transition-all hover:shadow-md">
                                                                        <div className="flex justify-between items-start mb-2">
                                                                            <h4 className="font-medium text-blue-900">{entry.question}</h4>
                                                                            <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                                                                                Canonical
                                                                            </span>
                                                                        </div>
                                                                        <p className="bg-white p-3 rounded border border-blue-100 text-gray-800">
                                                                            {entry.answer}
                                                                        </p>
                                                                        <div className="text-right mt-2 text-xs text-gray-500">
                                                                            Last updated: {entry.last_updated || 'Unknown'}
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Regular Questions */}
                                                {filteredEntries.some(e => !e.is_canonical) && (
                                                    <div>
                                                        <h3 className="font-medium text-gray-800 mb-3 flex items-center">
                                                            <span className="inline-block mr-2 w-2 h-2 rounded-full bg-purple-500"></span>
                                                            Regular Questions
                                                        </h3>
                                                        <div className="space-y-4">
                                                            {filteredEntries
                                                                .filter(entry => !entry.is_canonical)
                                                                .map((entry, i) => (
                                                                    <div key={i} className="bg-gray-50 rounded-lg p-4 border-l-4 border-gray-400 transition-all hover:shadow-md">
                                                                        <h4 className="font-medium text-gray-900 mb-2">{entry.question}</h4>
                                                                        <p className="bg-white p-3 rounded border border-gray-200 text-gray-800">
                                                                            {entry.answer}
                                                                        </p>
                                                                        <div className="text-right mt-2 text-xs text-gray-500">
                                                                            Last updated: {entry.last_updated || 'Unknown'}
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ) : (
                                            <div className="text-center py-8 text-gray-500 bg-gray-50 rounded-lg">
                                                {searchQuery ? (
                                                    <p>No questions match your search query "{searchQuery}".</p>
                                                ) : (
                                                    <p>No questions found in this cluster.</p>
                                                )}
                                            </div>
                                        )}

                                        <div className="mt-6 pt-4 border-t border-gray-200">
                                            <button
                                                className="w-full py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md transition-colors text-sm"
                                                onClick={() => setSelectedClusterId(null)}
                                            >
                                                Close
                                            </button>
                                        </div>
                                    </div>
                                )
                            ) : (
                                <div className="bg-gray-50 rounded-lg border border-gray-200 p-6 text-center h-64 flex items-center justify-center">
                                    <div className="text-gray-500">
                                        <h3 className="font-medium mb-2">No Cluster Selected</h3>
                                        <p>Select a cluster from the list to view its questions.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </MainLayout>
    );
}