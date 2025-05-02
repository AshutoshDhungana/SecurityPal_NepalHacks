//@ts-nocheck

'use client';

import { useState, useEffect } from 'react';
import MainLayout from '@/components/MainLayout';
import api from '@/lib/api';
import { Search } from 'lucide-react';
import { toast } from 'sonner';

export default function SimilaritySearchPage() {
    const [query, setQuery] = useState('');
    const [topK, setTopK] = useState(5);
    const [minSimilarity, setMinSimilarity] = useState(0.5);
    const [products, setProducts] = useState<any[]>([]);
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [categories, setCategories] = useState<string[]>([]);
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
    const [results, setResults] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);

    // Fetch products for dropdown
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

    // Fetch categories when product changes
    useEffect(() => {
        // Reset selected category when product changes
        setSelectedCategory(null);

        // Set categories regardless of product selection
        // This ensures categories are always available
        const availableCategories = [
            'API_Security',
            'Acceptable_Use_Policy',
            'Access_Control',
            'Anti-virus_and_Malware_Control',
            'Application',
            'Application_Security',
            'Artificial_Intelligence',
            'Asset_Management',
            'Audit_Assurance',
            'Backup_Policy_and_Procedures',
            'Business_Continuity_and_Disaster_Recovery',
            'Change_Control_and_Configuration_Management',
            'Data_Governance_and_Classification',
            'Data_Retention_and_Deletion',
            'Data_Security',
            'ESG',
            'Email_Security',
            'Encryption_and_Key_Management',
            'Hosting',
            'Human_Resources',
            'Identity_and_AccessManagement(IAM)',
            'Incident_Management',
            'Information_Security',
            'Integration',
            'Interoperability_and_Portability_Measures',
            'Legal',
            'Logging,_Monitoring_and_Audit_Trail',
            'Mobile_Device_Security',
            'Network_Security',
            'Organization',
            'Password_Policy_and_authentication_procedures',
            'Patch_Management',
            'Penetration_Testing',
            'Physical_Security_and_Environmental_Controls',
            'Privacy_and_Compliance',
            'Risk_Management',
            'Security_Awareness',
            'Server_Security',
            'Session_Management',
            'Software_Development_Life_Cycle',
            'Supply_Chain__Vendor_Management',
            'Vulnerability_and_Threat_Management',
            'Wireless_Security'
        ];
        setCategories(availableCategories);
    }, [selectedProduct]);

    // Simple mock function to encode a sentence to embedding 
    // (will be replaced by an actual API call in a production environment)
    function getMockEmbedding(text: string) {
        // Hash the text to generate a consistent pseudorandom seed
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }

        // Generate a pseudorandom embedding of dimension 384
        const embedding = [];
        // Use the hash as a seed
        let seed = hash;
        for (let i = 0; i < 384; i++) {
            // Simple LCG random number generator
            seed = (seed * 1664525 + 1013904223) % 4294967296;
            embedding.push((seed / 4294967296) * 2 - 1); // Value between -1 and 1
        }

        // Normalize the embedding
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        return embedding.map(val => val / magnitude);
    }

    // Calculate cosine similarity between two vectors
    function cosineSimilarity(vec1: number[], vec2: number[]) {
        let dotProduct = 0;
        let magnitude1 = 0;
        let magnitude2 = 0;

        for (let i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            magnitude1 += vec1[i] * vec1[i];
            magnitude2 += vec2[i] * vec2[i];
        }

        magnitude1 = Math.sqrt(magnitude1);
        magnitude2 = Math.sqrt(magnitude2);

        if (magnitude1 === 0 || magnitude2 === 0) return 0;

        return dotProduct / (magnitude1 * magnitude2);
    }

    // Search similar questions using the real API endpoint
    async function searchSimilarQuestions() {
        if (!query.trim()) {
            toast.error('Please enter a question to search');
            return;
        }

        setLoading(true);
        setResults([]);

        try {
            // Prepare search data - ensure product_id is always provided (even if empty string)
            const searchData = {
                query: query,
                product_id: selectedProduct || "", // Send empty string instead of undefined
                category: selectedCategory || undefined, // Only include if a category is selected
                threshold: minSimilarity,
                top_k: topK
            };

            console.log("Sending search request:", searchData); // Add logging for debugging

            // Use the real API endpoint instead of the mock implementation
            const response = await api.similaritySearch(searchData);

            // Check if we got results
            if (response && response.results) {
                // Filter out the embedding field from each result to avoid large data in UI
                const filteredResults = response.results.map(({ embedding, ...rest }) => rest);
                setResults(filteredResults);
            } else {
                // If no results or API error, fallback to mock implementation
                fallbackToMockSearch();
            }
        } catch (error) {
            console.error('Error searching for similar questions:', error);
            toast.error('Error searching for similar questions. Falling back to local search.');

            // Fallback to local mock implementation
            fallbackToMockSearch();
        } finally {
            setLoading(false);
        }
    }

    // Fallback function that uses the original mock implementation
    async function fallbackToMockSearch() {
        try {
            // Show warning toast
            toast.warning('Using local mock search - embeddings are not from real model');

            // Generate embedding for the query
            const queryEmbedding = getMockEmbedding(query);

            // Get clusters from API 
            const formattedProduct = selectedProduct ? selectedProduct.replace(' ', '_') : undefined;
            const params = {
                limit: 100, // Get more clusters to have enough questions
                min_size: 2,
                product: formattedProduct
            };

            const clusters = await api.getClusters(params);

            // Extract questions from clusters and add mock embeddings
            const questionsWithEmbeddings: any[] = [];

            clusters?.forEach((cluster: any) => {
                // Add canonical questions
                if (cluster.canonical_questions) {
                    cluster.canonical_questions.forEach((q: string) => {
                        if (q && typeof q === 'string') {
                            questionsWithEmbeddings.push({
                                question: q,
                                answer: `Canonical answer for cluster ${cluster.cluster_id}`,
                                embedding: getMockEmbedding(q),
                                is_canonical: true,
                                cluster_id: cluster.cluster_id
                            });
                        }
                    });
                }

                // Add regular questions (limit to first 3 per cluster to avoid processing too many)
                if (cluster.questions) {
                    cluster.questions.slice(0, 3).forEach((q: string) => {
                        if (q && typeof q === 'string') {
                            questionsWithEmbeddings.push({
                                question: q,
                                answer: `Regular answer for question in cluster ${cluster.cluster_id}`,
                                embedding: getMockEmbedding(q),
                                is_canonical: false,
                                cluster_id: cluster.cluster_id
                            });
                        }
                    });
                }
            });

            // If not enough questions, add some mock ones
            if (questionsWithEmbeddings.length < 20) {
                const mockQuestions = [
                    "How do I reset my password?",
                    "What security measures are in place for data protection?",
                    "How can I create a new account?",
                    "What is the pricing structure?",
                    "How do I export my data?",
                    "What browsers are supported?",
                    "How do I contact customer support?",
                    "What are the system requirements?",
                    "How do I change my notification settings?",
                    "Can I integrate with other platforms?"
                ];

                mockQuestions.forEach((q, i) => {
                    questionsWithEmbeddings.push({
                        question: q,
                        answer: `Answer to: ${q}`,
                        embedding: getMockEmbedding(q),
                        is_canonical: i % 3 === 0,
                        cluster_id: i % 5
                    });
                });
            }

            // Compute similarities
            const similarities = questionsWithEmbeddings.map(item => ({
                ...item,
                similarity: cosineSimilarity(queryEmbedding, item.embedding)
            }));

            // Sort by similarity and filter by min similarity
            const filteredResults = similarities
                .filter(item => item.similarity >= minSimilarity)
                .slice(0, topK)
                .map(({ embedding, ...rest }) => rest); // Remove the embedding field

            setResults(filteredResults);
        } catch (error) {
            console.error('Error in fallback search:', error);
            toast.error('Error in fallback search');
            setResults([]);
        }
    }

    return (
        <MainLayout>
            <div className="container mx-auto">
                <h1 className="text-3xl font-bold text-blue-600 mb-6">Sentence Similarity Search</h1>

                {/* Description */}
                <div className="bg-white rounded-lg shadow-md p-6 mb-8">
                    <p className="text-gray-600">
                        Enter a question or sentence to find similar questions in the database.
                        The system will calculate embeddings for your query and find the closest matches.
                    </p>
                </div>

                {/* Search Form */}
                <div className="bg-white rounded-lg shadow-md p-6 mb-8">
                    <div className="mb-6">
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Enter your question or sentence:
                        </label>
                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="e.g., How do I reset my password?"
                            className="w-full p-3 border border-gray-300 rounded-md resize-none h-32"
                        />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Number of results to show
                            </label>
                            <input
                                type="range"
                                min={1}
                                max={20}
                                value={topK}
                                onChange={(e) => setTopK(parseInt(e.target.value))}
                                className="w-full"
                            />
                            <div className="flex justify-between text-xs text-gray-500 mt-1">
                                <span>1</span>
                                <span>{topK}</span>
                                <span>20</span>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Minimum similarity threshold
                            </label>
                            <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.01}
                                value={minSimilarity}
                                onChange={(e) => setMinSimilarity(parseFloat(e.target.value))}
                                className="w-full"
                            />
                            <div className="flex justify-between text-xs text-gray-500 mt-1">
                                <span>0%</span>
                                <span>{(minSimilarity * 100).toFixed(0)}%</span>
                                <span>100%</span>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Search in specific product
                            </label>
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
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Filter by category
                            </label>
                            <select
                                value={selectedCategory || ''}
                                onChange={(e) => setSelectedCategory(e.target.value || null)}
                                className="w-full p-2 border border-gray-300 rounded-md"
                            >
                                <option value="">All Categories</option>
                                {categories.map(category => (
                                    <option key={category} value={category}>
                                        {category}
                                    </option>
                                ))}
                            </select>
                            {selectedProduct && categories.length === 0 && (
                                <p className="text-xs text-gray-500 mt-1">No categories available for this product</p>
                            )}
                            {!selectedProduct && (
                                <p className="text-xs text-gray-500 mt-1">Select a product first</p>
                            )}
                        </div>
                    </div>

                    <button
                        onClick={searchSimilarQuestions}
                        disabled={loading || !query.trim()}
                        className="w-full md:w-auto px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors flex items-center justify-center"
                    >
                        {loading ? (
                            <>
                                <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-r-2 border-white mr-2"></div>
                                Searching...
                            </>
                        ) : (
                            <>
                                <Search className="h-5 w-5 mr-2" />
                                Search Similar Questions
                            </>
                        )}
                    </button>
                </div>

                {/* Results */}
                {results.length > 0 && (
                    <div className="bg-white rounded-lg shadow-md p-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-6">
                            Found {results.length} similar questions
                        </h2>

                        <div className="space-y-6">
                            {results.map((result, index) => {
                                const similarityPercent = (result.similarity * 100).toFixed(1);
                                let similarityColor = '#F44336'; // Red for low similarity

                                if (result.similarity > 0.8) {
                                    similarityColor = '#4CAF50'; // Green for high similarity
                                } else if (result.similarity > 0.6) {
                                    similarityColor = '#FF9800'; // Orange for medium similarity
                                }

                                return (
                                    <div
                                        key={index}
                                        className={`
                      rounded-lg p-4 border-l-4
                      ${result.is_canonical ?
                                                'bg-blue-50 border-blue-600' :
                                                'bg-gray-50 border-gray-400'}
                    `}
                                    >
                                        <div className="flex justify-between mb-3">
                                            <h3 className={`font-medium ${result.is_canonical ? 'text-blue-900' : 'text-gray-900'}`}>
                                                {result.question}
                                            </h3>
                                            <span style={{ color: similarityColor }} className="font-medium">
                                                {similarityPercent}% Match
                                            </span>
                                        </div>

                                        <div className={`
                      p-3 rounded border
                      ${result.is_canonical ?
                                                'bg-white border-blue-100 text-gray-800' :
                                                'bg-white border-gray-200 text-gray-800'}
                    `}>
                                            {result.answer}
                                        </div>

                                        {result.is_canonical && (
                                            <div className="flex justify-end mt-2">
                                                <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                                                    Canonical
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}

                {loading && results.length === 0 && (
                    <div className="bg-white rounded-lg shadow-md p-6 text-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-r-2 border-blue-600 mx-auto mb-4"></div>
                        <p>Computing embeddings and searching for similar questions...</p>
                    </div>
                )}

                {!loading && query && results.length === 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-lg">
                        <p>No similar questions found. Try a different query or lower the similarity threshold.</p>
                    </div>
                )}
            </div>
        </MainLayout>
    );
}