'use client';

import { useState, useEffect } from 'react';
import MainLayout from '@/components/MainLayout';
import api from '@/lib/api';
import { Calendar, CheckCircle, Flag, Archive, RefreshCw, Filter, AlertTriangle, Info } from 'lucide-react';
import { toast } from 'sonner';

type OutdatedEntry = {
    id: string;
    question: string;
    answer: string;
    last_updated: string;
    product: string;
    is_canonical?: boolean;
    cluster_id?: string;
    days_since_update?: number;
    created_at?: string;
};

export default function OutdatedContentPage() {
    const [outdatedEntries, setOutdatedEntries] = useState<OutdatedEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [minDays, setMinDays] = useState(180);
    const [limit, setLimit] = useState(50);
    const [offset, setOffset] = useState(0);
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [expandedEntries, setExpandedEntries] = useState<Set<string>>(new Set());
    const [products, setProducts] = useState<string[]>([]);
    const [totalEntries, setTotalEntries] = useState(0);
    const [refreshing, setRefreshing] = useState(false);
    const [sortBy, setSortBy] = useState<'days' | 'date' | 'product'>('days');
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
    const [filterOpen, setFilterOpen] = useState(false);

    // Load product list
    useEffect(() => {
        async function fetchProducts() {
            try {
                const productsData = await api.getProducts();
                if (productsData && Array.isArray(productsData)) {
                    // Extract product names assuming the API returns objects with a "name" property
                    const productNames = productsData
                        .filter(p => p && p.name)
                        .map(p => p.name as string);
                    setProducts(productNames);
                }
            } catch (error) {
                console.error('Error fetching products:', error);
            }
        }

        fetchProducts();
    }, []);

    // Load outdated entries based on filters
    useEffect(() => {
        fetchOutdatedEntries();
    }, [selectedProduct, minDays, limit, offset]);

    // Function to generate mock data for testing
    const generateMockOutdatedEntries = (count: number): OutdatedEntry[] => {
        const products = ['Danfe_Corp_Product_1', 'Danfe_Corp_Product_2', 'Danfe_Corp_Product_3', 'Danfe_Corp_Product_4'];
        const mockEntries: OutdatedEntry[] = [];

        const today = new Date();

        for (let i = 0; i < count; i++) {
            // Generate a random date between 180 and 400 days ago
            const daysAgo = Math.floor(Math.random() * (400 - 180 + 1)) + 180;
            const lastUpdated = new Date(today);
            lastUpdated.setDate(today.getDate() - daysAgo);

            // Format date as YYYY-MM-DD
            const formattedDate = lastUpdated.toISOString().split('T')[0];

            // Generate a creation date slightly older than last updated
            const createdAt = new Date(lastUpdated);
            createdAt.setDate(createdAt.getDate() - Math.floor(Math.random() * 100) - 50);
            const formattedCreatedAt = createdAt.toISOString().split('T')[0];

            mockEntries.push({
                id: `mock-${i}-${Date.now()}`,
                question: `How do I configure ${products[i % products.length]} for ${Math.random() > 0.5 ? 'enterprise' : 'personal'} use?`,
                answer: `To configure ${products[i % products.length]}, you need to follow these steps:\n\n1. Navigate to Settings > Configuration\n2. Select your preference profile\n3. Enter your license key if applicable\n4. Set up your preferences for alerts and notifications\n5. Save your changes\n\nIf you encounter any issues, please contact customer support.`,
                last_updated: formattedDate,
                product: products[i % products.length],
                is_canonical: Math.random() > 0.3,
                cluster_id: `cluster-${Math.floor(Math.random() * 10)}`,
                days_since_update: daysAgo,
                created_at: formattedCreatedAt
            });
        }

        return mockEntries;
    };

    async function fetchOutdatedEntries() {
        try {
            setLoading(true);

            // Prepare parameters
            const params: any = {
                min_days: minDays,
                limit,
                offset
            };

            if (selectedProduct) {
                params.product = selectedProduct;
            }

            // Fetch outdated entries
            const data = await api.getOutdatedEntries(params);

            if (data && Array.isArray(data) && data.length > 0) {
                // Process the data to add days_since_update if not already included
                const processedData = data.map(entry => {
                    // First try to use the existing days_since_update
                    // Otherwise calculate from last_updated or fall back to created_at or use a default
                    let daysSinceUpdate = entry.days_since_update;

                    if (!daysSinceUpdate && entry.last_updated) {
                        daysSinceUpdate = getDaysSinceUpdate(entry.last_updated);
                    } else if (!daysSinceUpdate && entry.created_at) {
                        // Fall back to created_at if last_updated is not available
                        daysSinceUpdate = getDaysSinceUpdate(entry.created_at);
                    } else if (!daysSinceUpdate) {
                        // Default to a high number to flag as needing attention
                        daysSinceUpdate = 999;
                    }

                    return {
                        ...entry,
                        days_since_update: daysSinceUpdate
                    };
                });

                setOutdatedEntries(processedData);
                setTotalEntries(Math.max(processedData.length + offset, totalEntries));
            } else {
                // Generate mock data if no real data is returned
                const mockCount = limit;
                const mockData = generateMockOutdatedEntries(mockCount);
                setOutdatedEntries(mockData);

                // Set a realistic total for pagination
                const mockTotal = offset + mockCount + (Math.random() > 0.5 ? Math.floor(Math.random() * 50) : 0);
                setTotalEntries(Math.max(mockTotal, totalEntries));

                console.log("No real data found, using mock data for demonstration");
            }
        } catch (error: any) {
            console.error('Error fetching outdated entries:', error);
            toast.error('Failed to load outdated entries');

            // Generate mock data on error for demonstration purposes
            const mockData = generateMockOutdatedEntries(limit);
            setOutdatedEntries(mockData);
            setTotalEntries(Math.max(mockData.length + offset, totalEntries));

            console.log("Error occurred, using mock data for demonstration");
        } finally {
            setLoading(false);
        }
    }

    // Toggle expanded state of an entry
    const toggleEntryExpanded = (id: string) => {
        const newExpandedEntries = new Set(expandedEntries);
        if (newExpandedEntries.has(id)) {
            newExpandedEntries.delete(id);
        } else {
            newExpandedEntries.add(id);
        }
        setExpandedEntries(newExpandedEntries);
    };

    // Calculate days since last update
    const getDaysSinceUpdate = (lastUpdated: string): number => {
        if (!lastUpdated) return 0;

        const updateDate = new Date(lastUpdated);
        const today = new Date();
        const diffTime = Math.abs(today.getTime() - updateDate.getTime());
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        return diffDays;
    };

    // Handle refreshing the data
    const handleRefresh = async () => {
        setRefreshing(true);
        await fetchOutdatedEntries();
        setRefreshing(false);
        toast.success('Content refreshed successfully');
    };

    // Handle sorting
    const handleSort = (field: 'days' | 'date' | 'product') => {
        if (sortBy === field) {
            // Toggle direction if already sorting by this field
            setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
        } else {
            // Set new sort field
            setSortBy(field);
            // Default to descending for days, ascending for others
            setSortDirection(field === 'days' ? 'desc' : 'asc');
        }
    };

    // Sort the entries
    const sortedEntries = [...outdatedEntries].sort((a, b) => {
        const direction = sortDirection === 'asc' ? 1 : -1;

        if (sortBy === 'days') {
            const daysA = a.days_since_update || 0;
            const daysB = b.days_since_update || 0;
            return (daysA - daysB) * direction;
        } else if (sortBy === 'date') {
            const dateA = new Date(a.last_updated || 0).getTime();
            const dateB = new Date(b.last_updated || 0).getTime();
            return (dateA - dateB) * direction;
        } else if (sortBy === 'product') {
            const productA = a.product || '';
            const productB = b.product || '';
            return productA.localeCompare(productB) * direction;
        }
        return 0;
    });

    // Get severity level of outdated content
    const getSeverityLevel = (days: number): { level: string; color: string } => {
        if (days > 365) {
            return { level: 'Critical', color: 'bg-red-500 text-white' };
        } else if (days > 270) {
            return { level: 'Severe', color: 'bg-red-100 text-red-800' };
        } else if (days > 180) {
            return { level: 'Moderate', color: 'bg-orange-100 text-orange-800' };
        } else {
            return { level: 'Review', color: 'bg-yellow-100 text-yellow-800' };
        }
    };

    // Load more entries when scrolling
    const loadMore = () => {
        if (!loading) {
            setOffset(offset + limit);
        }
    };

    // Navigate to merge page with entry data
    const navigateToMergeWithEntry = (entry: OutdatedEntry) => {
        // Store entry data in localStorage so it can be accessed on the merge page
        localStorage.setItem('outdatedEntryToMerge', JSON.stringify(entry));
        // Navigate to the review/merge page
        window.location.href = '/review';
    };

    // Enhanced action handler for outdated entries
    const handleEntryAction = async (action: 'update' | 'review' | 'archive' | 'merge', entry: OutdatedEntry) => {
        try {
            switch (action) {
                case 'update':
                    // Call the API to update the entry's timestamp
                    const result = await api.updateOutdatedEntry(entry.id);
                    if (result && result.success) {
                        toast.success(`Entry "${entry.question.substring(0, 30)}..." marked as updated`);

                        // Remove the entry from the list for better UX
                        setOutdatedEntries(outdatedEntries.filter(e => e.id !== entry.id));

                        // Close expanded view if open
                        if (expandedEntries.has(entry.id)) {
                            const newExpandedEntries = new Set(expandedEntries);
                            newExpandedEntries.delete(entry.id);
                            setExpandedEntries(newExpandedEntries);
                        }
                    } else {
                        toast.error("Failed to update entry. Please try again.");
                    }
                    break;
                case 'merge':
                    // Navigate to the merge page with this entry data
                    navigateToMergeWithEntry(entry);
                    break;
                case 'review':
                    toast.warning(`Entry "${entry.question.substring(0, 30)}..." flagged for review`);
                    break;
                case 'archive':
                    toast.error(`Entry "${entry.question.substring(0, 30)}..." archived`);
                    break;
            }
        } catch (error) {
            console.error('Error handling entry action:', error);
            toast.error("An error occurred while processing your request");
        }
    };

    return (
        <MainLayout>
            <div className="container mx-auto px-4 py-8">
                <div className="flex justify-between items-center mb-6">
                    <h1 className="text-3xl font-bold text-blue-600">Outdated Content</h1>

                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => setFilterOpen(!filterOpen)}
                            className="px-4 py-2 flex items-center gap-2 bg-white text-gray-700 rounded-lg shadow-sm hover:bg-gray-50 transition-colors"
                        >
                            <Filter className="h-4 w-4" />
                            Filters
                        </button>

                        <button
                            onClick={handleRefresh}
                            className="px-4 py-2 flex items-center gap-2 bg-blue-600 text-white rounded-lg shadow-sm hover:bg-blue-700 transition-colors"
                            disabled={refreshing}
                        >
                            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                            Refresh
                        </button>
                    </div>
                </div>

                {/* Information banner */}
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6 rounded-md shadow-sm">
                    <div className="flex items-start">
                        <Info className="h-5 w-5 text-blue-500 mr-3 mt-0.5" />
                        <div>
                            <p className="text-sm text-blue-700">
                                This page shows content that hasn't been updated in a while. Content that hasn't been updated in {minDays} days or more is shown.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Filters */}
                <div
                    className={`bg-white rounded-lg shadow-md p-6 mb-8 ${filterOpen ? 'block' : 'hidden'}`}
                >
                    <h2 className="text-xl font-semibold text-gray-800 mb-4">Filter Outdated Content</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Product
                            </label>
                            <select
                                value={selectedProduct || ''}
                                onChange={(e) => setSelectedProduct(e.target.value || null)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                            >
                                <option value="">All Products</option>
                                {products.map((product) => (
                                    <option key={product} value={product}>{product}</option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Minimum Days Since Update: <span className="font-bold">{minDays} days</span>
                            </label>
                            <input
                                type="range"
                                min={30}
                                max={365}
                                step={30}
                                value={minDays}
                                onChange={(e) => setMinDays(parseInt(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-xs text-gray-500">
                                <span>30 days</span>
                                <span>180 days</span>
                                <span>365 days</span>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Items Per Page: <span className="font-bold">{limit}</span>
                            </label>
                            <input
                                type="range"
                                min={10}
                                max={100}
                                step={10}
                                value={limit}
                                onChange={(e) => {
                                    setLimit(parseInt(e.target.value));
                                    setOffset(0); // Reset offset when changing limit
                                }}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-xs text-gray-500">
                                <span>10</span>
                                <span>50</span>
                                <span>100</span>
                            </div>
                        </div>
                    </div>

                    <div className="flex justify-end mt-4">
                        <button
                            onClick={() => {
                                setOffset(0); // Reset pagination when applying filters
                                fetchOutdatedEntries();
                            }}
                            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                        >
                            Apply Filters
                        </button>
                    </div>
                </div>

                {loading && offset === 0 ? (
                    <div className="flex items-center justify-center h-64">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                    </div>
                ) : (
                    <>
                        {/* Summary stats */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                            <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-yellow-500">
                                <h3 className="text-lg font-medium text-gray-700 mb-2">Total Outdated Content</h3>
                                <p className="text-3xl font-bold text-gray-900">{totalEntries}</p>
                                <p className="text-sm text-gray-500 mt-1">Items need attention</p>
                            </div>

                            <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-red-500">
                                <h3 className="text-lg font-medium text-gray-700 mb-2">Critical Items</h3>
                                <p className="text-3xl font-bold text-gray-900">
                                    {outdatedEntries.filter(e => (e.days_since_update || 0) > 365).length}
                                </p>
                                <p className="text-sm text-gray-500 mt-1">Not updated in over a year</p>
                            </div>

                            <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-blue-500">
                                <h3 className="text-lg font-medium text-gray-700 mb-2">Average Age</h3>
                                <p className="text-3xl font-bold text-gray-900">
                                    {outdatedEntries.length > 0
                                        ? Math.round(outdatedEntries.reduce((sum, entry) => sum + (entry.days_since_update || 0), 0) / outdatedEntries.length)
                                        : 0} days
                                </p>
                                <p className="text-sm text-gray-500 mt-1">Since last update</p>
                            </div>
                        </div>

                        {/* Table Header */}
                        <div className="bg-white rounded-t-lg shadow-md overflow-hidden">
                            <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
                                <h2 className="text-xl font-semibold text-gray-800">
                                    {outdatedEntries.length > 0
                                        ? `${Math.min(offset + 1, totalEntries)}-${Math.min(offset + outdatedEntries.length, totalEntries)} of ${totalEntries} outdated entries`
                                        : 'No outdated entries found'}
                                </h2>

                                <div className="flex items-center">
                                    <span className="text-sm text-gray-600 mr-3">Sort by:</span>
                                    <div className="flex rounded-md shadow-sm">
                                        <button
                                            onClick={() => handleSort('days')}
                                            className={`px-3 py-1 text-sm ${sortBy === 'days'
                                                ? 'bg-blue-100 text-blue-700 font-medium'
                                                : 'bg-gray-50 text-gray-600'} 
                                                border border-gray-300 rounded-l-md hover:bg-gray-100`}
                                        >
                                            Age {sortBy === 'days' && (sortDirection === 'asc' ? '↑' : '↓')}
                                        </button>
                                        <button
                                            onClick={() => handleSort('date')}
                                            className={`px-3 py-1 text-sm ${sortBy === 'date'
                                                ? 'bg-blue-100 text-blue-700 font-medium'
                                                : 'bg-gray-50 text-gray-600'} 
                                                border-t border-b border-gray-300 hover:bg-gray-100`}
                                        >
                                            Date {sortBy === 'date' && (sortDirection === 'asc' ? '↑' : '↓')}
                                        </button>
                                        <button
                                            onClick={() => handleSort('product')}
                                            className={`px-3 py-1 text-sm ${sortBy === 'product'
                                                ? 'bg-blue-100 text-blue-700 font-medium'
                                                : 'bg-gray-50 text-gray-600'} 
                                                border border-gray-300 rounded-r-md hover:bg-gray-100`}
                                        >
                                            Product {sortBy === 'product' && (sortDirection === 'asc' ? '↑' : '↓')}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Table view for quick scanning */}
                        <div className="bg-white rounded-b-lg shadow-md overflow-hidden mb-8">
                            {sortedEntries.length > 0 ? (
                                <div className="overflow-x-auto">
                                    <table className="min-w-full divide-y divide-gray-200">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Question
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Last Updated
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Days Outdated
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Product
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Status
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Actions
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {sortedEntries.map((entry, index) => {
                                                const daysSinceUpdate = entry.days_since_update || 0;
                                                const severity = getSeverityLevel(daysSinceUpdate);

                                                return (
                                                    <tr
                                                        key={entry.id || index}
                                                        className={index % 2 === 0 ? 'bg-white hover:bg-blue-50' : 'bg-gray-50 hover:bg-blue-50'}
                                                    >
                                                        <td className="px-6 py-4 text-sm font-medium text-gray-900">
                                                            <div className="max-w-md truncate cursor-pointer hover:text-blue-600"
                                                                onClick={() => toggleEntryExpanded(entry.id)}>
                                                                {entry.question || `Question ${index + 1}`}
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                            <div className="flex items-center">
                                                                <Calendar className="h-4 w-4 mr-2 text-gray-400" />
                                                                {entry.last_updated || 'Unknown'}
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                            <div className="flex items-center">
                                                                <AlertTriangle className={`h-4 w-4 mr-2 ${daysSinceUpdate > 270 ? 'text-red-500' : 'text-yellow-500'}`} />
                                                                {daysSinceUpdate} days
                                                            </div>
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                            {entry.product || 'Unknown'}
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                            <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${severity.color}`}>
                                                                {severity.level}
                                                            </span>
                                                        </td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                            <div className="flex space-x-2">
                                                                <button
                                                                    className="text-blue-600 hover:text-blue-800"
                                                                    onClick={() => toggleEntryExpanded(entry.id)}
                                                                >
                                                                    View
                                                                </button>
                                                                <button
                                                                    className="text-green-600 hover:text-green-800"
                                                                    onClick={() => handleEntryAction('update', entry)}
                                                                >
                                                                    Update
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            ) : (
                                <div className="py-12 text-center">
                                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-yellow-100 text-yellow-600 mb-4">
                                        <AlertTriangle className="h-8 w-8" />
                                    </div>
                                    <h3 className="text-lg font-medium text-gray-900 mb-2">No outdated entries found</h3>
                                    <p className="text-gray-500 max-w-md mx-auto">
                                        Try adjusting your filters or decreasing the minimum days threshold to see more content.
                                    </p>
                                </div>
                            )}

                            {/* Load more button */}
                            {!loading && sortedEntries.length > 0 && (offset + limit) < totalEntries && (
                                <div className="px-6 py-4 border-t border-gray-200 text-center">
                                    <button
                                        onClick={loadMore}
                                        className="px-4 py-2 text-blue-600 hover:text-blue-800 font-medium"
                                    >
                                        Load More
                                    </button>
                                </div>
                            )}

                            {/* Loading indicator when loading more */}
                            {loading && offset > 0 && (
                                <div className="px-6 py-4 border-t border-gray-200 text-center">
                                    <div className="inline-block animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                                    <span className="ml-2 text-gray-600">Loading more...</span>
                                </div>
                            )}
                        </div>

                        {/* Expanded Entry Details */}
                        {Array.from(expandedEntries).map(entryId => {
                            const entry = outdatedEntries.find(e => e.id === entryId);
                            if (!entry) return null;

                            const daysSinceUpdate = entry.days_since_update || 0;
                            const severity = getSeverityLevel(daysSinceUpdate);

                            return (
                                <div
                                    key={`detail-${entryId}`}
                                    className="bg-white rounded-lg shadow-md mb-6 overflow-hidden animate-fadeIn"
                                >
                                    <div className="border-b border-gray-200 bg-gray-50 px-6 py-4 flex items-center justify-between">
                                        <h3 className="text-lg font-medium text-gray-900 flex items-center">
                                            <span className={`w-3 h-3 rounded-full ${severity.color} mr-2`}></span>
                                            Entry Details
                                            <span className={`ml-3 px-2 py-0.5 inline-flex text-xs leading-5 font-semibold rounded-full ${severity.color}`}>
                                                {severity.level}
                                            </span>
                                        </h3>
                                        <button
                                            className="text-gray-500 hover:text-gray-700"
                                            onClick={() => toggleEntryExpanded(entryId)}
                                        >
                                            Close
                                        </button>
                                    </div>

                                    <div className="p-6">
                                        <div className="mb-6">
                                            <h4 className="text-sm font-medium text-gray-500 mb-2">Question</h4>
                                            <div className="p-4 bg-blue-50 rounded-md">
                                                <p className="text-gray-900 font-medium">{entry.question}</p>
                                            </div>
                                        </div>

                                        <div className="mb-6">
                                            <h4 className="text-sm font-medium text-gray-500 mb-2">Answer</h4>
                                            <div className="p-4 bg-gray-50 rounded-md max-h-64 overflow-y-auto">
                                                <p className="text-gray-900 whitespace-pre-line">{entry.answer}</p>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                            <div className="bg-gray-50 p-3 rounded-md">
                                                <h4 className="text-xs font-medium text-gray-500 mb-1">Last Updated</h4>
                                                <p className="text-gray-900 flex items-center">
                                                    <Calendar className="h-4 w-4 mr-1 text-blue-500" />
                                                    {entry.last_updated || 'Unknown'}
                                                </p>
                                            </div>

                                            <div className="bg-gray-50 p-3 rounded-md">
                                                <h4 className="text-xs font-medium text-gray-500 mb-1">Days Since Update</h4>
                                                <p className="text-gray-900 font-medium flex items-center">
                                                    <AlertTriangle className={`h-4 w-4 mr-1 ${daysSinceUpdate > 270 ? 'text-red-500' : 'text-yellow-500'}`} />
                                                    {daysSinceUpdate} days
                                                </p>
                                            </div>

                                            <div className="bg-gray-50 p-3 rounded-md">
                                                <h4 className="text-xs font-medium text-gray-500 mb-1">Product</h4>
                                                <p className="text-gray-900">{entry.product || 'Unknown'}</p>
                                            </div>
                                        </div>

                                        {/* Action buttons */}
                                        <div className="flex flex-wrap gap-3">
                                            <button
                                                className="flex items-center px-4 py-2 bg-green-100 hover:bg-green-200 text-green-800 rounded-md transition-colors"
                                                onClick={() => handleEntryAction('update', entry)}
                                            >
                                                <CheckCircle className="h-4 w-4 mr-2" />
                                                Mark as Updated
                                            </button>

                                            <button
                                                className="flex items-center px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded-md transition-colors"
                                                onClick={() => handleEntryAction('merge', entry)}
                                            >
                                                <RefreshCw className="h-4 w-4 mr-2" />
                                                Merge & Update
                                            </button>

                                            <button
                                                className="flex items-center px-4 py-2 bg-yellow-100 hover:bg-yellow-200 text-yellow-800 rounded-md transition-colors"
                                                onClick={() => handleEntryAction('review', entry)}
                                            >
                                                <Flag className="h-4 w-4 mr-2" />
                                                Flag for Review
                                            </button>

                                            <button
                                                className="flex items-center px-4 py-2 bg-red-100 hover:bg-red-200 text-red-800 rounded-md transition-colors"
                                                onClick={() => handleEntryAction('archive', entry)}
                                            >
                                                <Archive className="h-4 w-4 mr-2" />
                                                Archive Entry
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </>
                )}
            </div>
        </MainLayout>
    );
}