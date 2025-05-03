'use client';

import { useState, useEffect } from 'react';
import MainLayout from '@/components/MainLayout';
import api from '@/lib/api';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CornerDownLeft, Save, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';

export default function ReviewPanelPage() {
    // State for QA pair comparison and merging
    const [qaPair1, setQaPair1] = useState({
        id: "qa1",
        question: "What are the security measures in place for data access?",
        answer: "Our system implements role-based access control (RBAC) to manage data access permissions. All data requests are logged and audited. We use encryption for data at rest and in transit, and employ multi-factor authentication for sensitive operations.",
        last_updated: "2022-05-15"
    });

    const [qaPair2, setQaPair2] = useState({
        id: "qa2",
        question: "How does your system control access to data?",
        answer: "We use role-based access control to manage permissions. All access is logged. Data is encrypted both at rest and in transit, and we require multi-factor authentication for sensitive operations.",
        last_updated: "2023-01-10"
    });

    const [mergedPair, setMergedPair] = useState<any>(null);
    const [mergedHistory, setMergedHistory] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [fetchingHistory, setFetchingHistory] = useState(false);
    const [calculatingSimilarity, setCalculatingSimilarity] = useState(false);
    const [similarityScore, setSimilarityScore] = useState<number | null>(null);

    // Fetch merged history on page load
    useEffect(() => {
        fetchMergedHistory();
    }, []);

    // Check for outdated entry in localStorage on component mount
    useEffect(() => {
        // Check if there's an outdated entry in localStorage
        const outdatedEntryJson = localStorage.getItem('outdatedEntryToMerge');
        if (outdatedEntryJson) {
            try {
                const outdatedEntry = JSON.parse(outdatedEntryJson);
                // Set the outdated entry as the first QA pair
                setQaPair1({
                    id: outdatedEntry.id,
                    question: outdatedEntry.question,
                    answer: outdatedEntry.answer,
                    last_updated: outdatedEntry.last_updated || "Unknown"
                });

                // Clear the localStorage item to avoid reusing it on page refresh
                localStorage.removeItem('outdatedEntryToMerge');

                // Show a toast notification
                toast.info("Outdated entry loaded for merging");
            } catch (error) {
                console.error("Error parsing outdated entry:", error);
            }
        }
    }, []);

    // Fetch merged QA pairs history
    const fetchMergedHistory = async () => {
        setFetchingHistory(true);
        try {
            const history = await api.getMergedQAPairs();
            setMergedHistory(history || []);
        } catch (error) {
            console.error('Error fetching merged history:', error);
            toast.error('Failed to load merged QA pairs history');
            // Generate mock history if API fails
            generateMockMergedHistory();
        } finally {
            setFetchingHistory(false);
        }
    };

    // Generate mock history data
    const generateMockMergedHistory = () => {
        const mockHistory = [
            {
                id: 'merged_1',
                question: 'What are the security measures for protecting user data?',
                answer: 'We implement multiple security measures including role-based access control (RBAC), encryption of data both at rest and in transit, regular security audits, and multi-factor authentication for sensitive operations.',
                merged_at: '2023-04-15',
                sources: ['qa1', 'qa2'],
                is_canonical: true
            },
            {
                id: 'merged_2',
                question: 'How do I export my data from the platform?',
                answer: 'You can export your data from the Settings page. Navigate to Settings > Data, and use the Export options. Your data will be available in CSV or JSON format.',
                merged_at: '2023-03-22',
                sources: ['qa3', 'qa4'],
                is_canonical: true
            },
            {
                id: 'merged_3',
                question: 'What browsers are supported by the platform?',
                answer: 'Our platform supports all modern browsers including Chrome, Firefox, Safari, and Edge in their latest versions. Internet Explorer is not supported.',
                merged_at: '2023-02-10',
                sources: ['qa5', 'qa6'],
                is_canonical: false
            }
        ];

        setMergedHistory(mockHistory);
    };

    // Simple mock function to calculate similarity
    const calculateSimilarity = () => {
        setCalculatingSimilarity(true);

        // Simple mock implementation - would be replaced with actual embedding comparison
        const mockSimilarity = () => {
            const length1 = qaPair1.question.length + qaPair1.answer.length;
            const length2 = qaPair2.question.length + qaPair2.answer.length;

            // Convert both to lowercase for comparison
            const text1 = (qaPair1.question + qaPair1.answer).toLowerCase();
            const text2 = (qaPair2.question + qaPair2.answer).toLowerCase();

            // Count shared words
            const words1 = text1.split(/\s+/);
            const words2 = text2.split(/\s+/);
            const uniqueWords1 = new Set(words1);
            const uniqueWords2 = new Set(words2);

            let sharedWords = 0;
            uniqueWords1.forEach(word => {
                if (uniqueWords2.has(word)) sharedWords++;
            });

            // Combine text similarity with length similarity
            const lengthRatio = Math.min(length1, length2) / Math.max(length1, length2);
            const wordSimilarity = sharedWords / (uniqueWords1.size + uniqueWords2.size - sharedWords);

            return (lengthRatio * 0.3) + (wordSimilarity * 0.7);
        };

        // Simulate API delay
        setTimeout(() => {
            const score = mockSimilarity();
            setSimilarityScore(Math.min(Math.max(score, 0), 1));
            setCalculatingSimilarity(false);
        }, 1000);
    };

    // Merge QA pairs
    const mergeQAPairs = async () => {
        setLoading(true);
        try {
            // Prepare request data
            const mergeRequest = {
                pair1: {
                    id: qaPair1.id,
                    question: qaPair1.question,
                    answer: qaPair1.answer
                },
                pair2: {
                    id: qaPair2.id,
                    question: qaPair2.question,
                    answer: qaPair2.answer
                },
                user_id: "admin", // In a real app, get from authentication
                similarity_score: similarityScore || 0.5
            };

            // Use API to merge
            try {
                const result = await api.mergeQAPairs(mergeRequest);
                setMergedPair(result);
                toast.success("QA pairs merged successfully!");
            } catch (error) {
                console.error('Error from API:', error);
                // If API fails, create a mock merged result
                createMockMergedResult();
            }
        } catch (error) {
            console.error('Error merging QA pairs:', error);
            toast.error('Failed to merge QA pairs');
            // Create mock result if merge fails
            createMockMergedResult();
        } finally {
            setLoading(false);
        }
    };

    // Create a mock merged result
    const createMockMergedResult = () => {
        // Take the longer question
        const mergedQuestion = qaPair1.question.length > qaPair2.question.length
            ? qaPair1.question
            : qaPair2.question;

        // Combine the answers
        const mergedAnswer = `${qaPair1.answer} Additional information: ${qaPair2.answer}`;

        setMergedPair({
            id: `merged_${Date.now()}`,
            question: mergedQuestion,
            answer: mergedAnswer,
            merged_at: new Date().toISOString().split('T')[0],
            sources: [qaPair1.id, qaPair2.id]
        });

        toast.info("Using local merge function (API unavailable)");
    };

    // Save the merged QA pair
    const saveMergedPair = async () => {
        if (!mergedPair) return;

        setLoading(true);
        try {
            // Prepare save request
            const saveRequest = {
                id: mergedPair.id,
                question: mergedPair.question,
                answer: mergedPair.answer
            };

            // Use API to save
            try {
                const result = await api.saveMergedPair(saveRequest);
                toast.success("Merged QA pair saved successfully!");

                // After saving, refresh the merged history
                fetchMergedHistory();

                // Reset the QA pairs with new example data after successful save
                resetQAPairsWithNewData();

            } catch (error) {
                console.error('Error from API:', error);
                toast.info("Save simulation successful (API unavailable)");

                // Simulate adding to history with the merged pair
                setMergedHistory(prev => [
                    {
                        ...mergedPair,
                        merged_at: new Date().toISOString().split('T')[0]
                    },
                    ...prev
                ]);

                // Reset the QA pairs with new example data after simulation
                resetQAPairsWithNewData();
            }
        } catch (error) {
            console.error('Error saving merged QA pair:', error);
            toast.error('Failed to save merged QA pair');
        } finally {
            setLoading(false);
        }
    };

    // Reset the merge process
    const resetMerge = () => {
        setMergedPair(null);
        setSimilarityScore(null);
    };

    // Function to load new QA pairs after a successful merge
    const resetQAPairsWithNewData = () => {
        // Reset the merged state
        setMergedPair(null);
        setSimilarityScore(null);

        // Set new example QA pairs
        setQaPair1({
            id: `qa${Math.floor(Math.random() * 1000)}`,
            question: "How often are security patches applied to the system?",
            answer: "Security patches are applied monthly during our regular maintenance window. Critical security updates are applied within 24 hours of release following our expedited testing process.",
            last_updated: "2023-08-12"
        });

        setQaPair2({
            id: `qa${Math.floor(Math.random() * 1000)}`,
            question: "What is your security patching schedule?",
            answer: "We follow a monthly patching schedule for regular updates. However, for critical security vulnerabilities, we implement patches within 24 hours after appropriate testing.",
            last_updated: "2024-01-05"
        });
    };

    // Update the merged result
    const updateMergedField = (field: 'question' | 'answer', value: string) => {
        if (!mergedPair) return;

        setMergedPair((prev: any) => ({
            ...prev,
            [field]: value
        }));
    };

    return (
        <MainLayout>
            <div className="w-full">
                <h1 className="text-3xl font-bold text-blue-600 mb-6">Review Panel</h1>

                <Tabs defaultValue="merge" className="w-full">
                    <TabsList className="mb-4">
                        <TabsTrigger value="merge">Merge Items</TabsTrigger>
                        <TabsTrigger value="history">Merged History</TabsTrigger>
                    </TabsList>

                    <TabsContent value="merge">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                            {/* QA Pair 1 */}
                            <Card>
                                <CardHeader>
                                    <CardTitle>Entry 1</CardTitle>
                                    {/*  */}
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-4">
                                        <div>
                                            <h3 className="text-sm font-medium text-gray-500 mb-1">Question</h3>
                                            <textarea
                                                value={qaPair1.question}
                                                onChange={e => setQaPair1({ ...qaPair1, question: e.target.value })}
                                                className="w-full p-2 border border-gray-200 rounded-md min-h-[100px]"
                                            />
                                        </div>

                                        <div>
                                            <h3 className="text-sm font-medium text-gray-500 mb-1">Answer</h3>
                                            <textarea
                                                value={qaPair1.answer}
                                                onChange={e => setQaPair1({ ...qaPair1, answer: e.target.value })}
                                                className="w-full p-2 border border-gray-200 rounded-md min-h-[150px]"
                                            />
                                        </div>

                                        <div>
                                            <p className="text-sm text-gray-500">
                                                Last Updated: {qaPair1.last_updated}
                                            </p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* QA Pair 2 */}
                            <Card>
                                <CardHeader>
                                    <CardTitle>Entry 2</CardTitle>
                                    {/* <CardDescription>ID: {qaPair2.id}</CardDescription> */}
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-4">
                                        <div>
                                            <h3 className="text-sm font-medium text-gray-500 mb-1">Question</h3>
                                            <textarea
                                                value={qaPair2.question}
                                                onChange={e => setQaPair2({ ...qaPair2, question: e.target.value })}
                                                className="w-full p-2 border border-gray-200 rounded-md min-h-[100px]"
                                            />
                                        </div>

                                        <div>
                                            <h3 className="text-sm font-medium text-gray-500 mb-1">Answer</h3>
                                            <textarea
                                                value={qaPair2.answer}
                                                onChange={e => setQaPair2({ ...qaPair2, answer: e.target.value })}
                                                className="w-full p-2 border border-gray-200 rounded-md min-h-[150px]"
                                            />
                                        </div>

                                        <div>
                                            <p className="text-sm text-gray-500">
                                                Last Updated: {qaPair2.last_updated}
                                            </p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Similarity Analysis */}
                        <Card className="mb-8">
                            <CardHeader>
                                <CardTitle>Similarity Analysis</CardTitle>
                                <CardDescription>
                                    Analyze the similarity between the two entries
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div>
                                        {similarityScore !== null ? (
                                            <div className="flex items-center">
                                                <Badge
                                                    className={`
                            px-3 py-1 text-base
                            ${similarityScore >= 0.8 ? 'bg-green-100 text-green-800' :
                                                            similarityScore >= 0.5 ? 'bg-yellow-100 text-yellow-800' :
                                                                'bg-red-100 text-red-800'}
                          `}
                                                >
                                                    {(similarityScore * 100).toFixed(1)}% Similar
                                                </Badge>
                                                <p className="ml-4 text-gray-500">
                                                    {similarityScore >= 0.8 ? 'High similarity - good merge candidate' :
                                                        similarityScore >= 0.5 ? 'Medium similarity - review carefully' :
                                                            'Low similarity - consider keeping separate'}
                                                </p>
                                            </div>
                                        ) : (
                                            <p className="text-gray-500">
                                                Click "Calculate Similarity" to analyze these entries
                                            </p>
                                        )}
                                    </div>
                                    <Button
                                        onClick={calculateSimilarity}
                                        disabled={calculatingSimilarity}
                                        variant="outline"
                                    >
                                        {calculatingSimilarity ? (
                                            <>
                                                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                                Calculating...
                                            </>
                                        ) : (
                                            'Calculate Similarity'
                                        )}
                                    </Button>
                                </div>

                                <div className="mt-4 pt-4 border-t border-gray-200">
                                    <h3 className="font-medium mb-2">Key Differences</h3>
                                    <ul className="list-disc list-inside space-y-1 text-gray-600">
                                        <li>Different question phrasing but similar intent</li>
                                        <li>Second answer is more concise</li>
                                        <li>Both mention role-based access control and encryption</li>
                                    </ul>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Merged Result or Decision Buttons */}
                        {mergedPair ? (
                            <Card>
                                <CardHeader>
                                    <CardTitle>Merged Result</CardTitle>
                                    <CardDescription>
                                        Edit the merged result before saving
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-4">
                                        <div>
                                            <h3 className="text-sm font-medium text-gray-500 mb-1">Merged Question</h3>
                                            <textarea
                                                value={mergedPair.question}
                                                onChange={e => updateMergedField('question', e.target.value)}
                                                className="w-full p-2 border border-gray-200 rounded-md min-h-[100px]"
                                            />
                                        </div>

                                        <div>
                                            <h3 className="text-sm font-medium text-gray-500 mb-1">Merged Answer</h3>
                                            <textarea
                                                value={mergedPair.answer}
                                                onChange={e => updateMergedField('answer', e.target.value)}
                                                className="w-full p-2 border border-gray-200 rounded-md min-h-[150px]"
                                            />
                                        </div>

                                        {mergedPair.merged_at && (
                                            <div>
                                                <p className="text-sm text-gray-500">
                                                    Merged At: {mergedPair.merged_at}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                </CardContent>
                                <CardFooter className="flex justify-between">
                                    <Button variant="outline" onClick={resetMerge}>
                                        Reset
                                    </Button>
                                    <Button onClick={saveMergedPair} disabled={loading}>
                                        {loading ? (
                                            <>
                                                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                                Saving...
                                            </>
                                        ) : (
                                            <>
                                                <Save className="mr-2 h-4 w-4" />
                                                Save Merged Result
                                            </>
                                        )}
                                    </Button>
                                </CardFooter>
                            </Card>
                        ) : (
                            <Card>
                                <CardHeader>
                                    <CardTitle>Decision</CardTitle>
                                    <CardDescription>
                                        Choose how to handle these entries
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                        <Button
                                            className="w-full h-20 bg-green-600 hover:bg-green-700"
                                            onClick={mergeQAPairs}
                                            disabled={loading}
                                        >
                                            {loading ? (
                                                <>
                                                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                                    Merging...
                                                </>
                                            ) : (
                                                'Merge Entries'
                                            )}
                                        </Button>

                                        <Button
                                            className="w-full h-20"
                                            variant="outline"
                                            onClick={() => toast.info('Both entries will be kept')}
                                        >
                                            Keep Both
                                        </Button>

                                        <Button
                                            className="w-full h-20 bg-blue-600 hover:bg-blue-700"
                                            onClick={mergeQAPairs}
                                        >
                                            Edit & Merge
                                        </Button>

                                        <Button
                                            className="w-full h-20"
                                            variant="outline"
                                            onClick={() => toast.warning('Decision deferred')}
                                        >
                                            Defer Decision
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        )}
                    </TabsContent>

                    <TabsContent value="history">
                        <div className="bg-white rounded-lg shadow-md p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-xl font-semibold text-gray-800">
                                    Previously Merged QA Pairs
                                </h2>
                                <Button
                                    variant="outline"
                                    onClick={fetchMergedHistory}
                                    disabled={fetchingHistory}
                                >
                                    {fetchingHistory ? (
                                        <>
                                            <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                            Refreshing...
                                        </>
                                    ) : (
                                        <>
                                            <RefreshCw className="mr-2 h-4 w-4" />
                                            Refresh
                                        </>
                                    )}
                                </Button>
                            </div>

                            {fetchingHistory ? (
                                <div className="flex items-center justify-center h-64">
                                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                                </div>
                            ) : mergedHistory.length > 0 ? (
                                <div className="space-y-6">
                                    {mergedHistory.map((pair, index) => (
                                        <div
                                            key={pair.id || index}
                                            className="border border-gray-200 rounded-lg overflow-hidden"
                                        >
                                            <div className="p-4 cursor-pointer bg-gray-50 flex justify-between items-center"
                                                onClick={() => document.getElementById(`content-${index}`)?.classList.toggle('hidden')}
                                            >
                                                <h3 className="font-medium text-gray-900">
                                                    {pair.question.length > 70
                                                        ? `${pair.question.substring(0, 70)}...`
                                                        : pair.question}
                                                </h3>
                                                <div>
                                                    <CornerDownLeft className="h-4 w-4 text-gray-400" />
                                                </div>
                                            </div>

                                            <div id={`content-${index}`} className="p-4 hidden">
                                                <div className="mb-4">
                                                    <h4 className="text-sm font-medium text-gray-500 mb-1">Question</h4>
                                                    <p className="text-gray-900">{pair.question}</p>
                                                </div>

                                                <div className="mb-4">
                                                    <h4 className="text-sm font-medium text-gray-500 mb-1">Answer</h4>
                                                    <div className="p-3 bg-gray-50 rounded">
                                                        <p className="text-gray-900">{pair.answer}</p>
                                                    </div>
                                                </div>

                                                <div className="grid grid-cols-2 gap-4">
                                                    <div>
                                                        <h4 className="text-sm font-medium text-gray-500 mb-1">Merged At</h4>
                                                        <p className="text-gray-900">{pair.merged_at || 'Unknown'}</p>
                                                    </div>

                                                    <div>
                                                        <h4 className="text-sm font-medium text-gray-500 mb-1">Sources</h4>
                                                        <p className="text-gray-900">
                                                            {pair.sources ? pair.sources.join(', ') : 'Unknown'}
                                                        </p>
                                                    </div>
                                                </div>

                                                {pair.is_canonical !== undefined && (
                                                    <div className="mt-4 pt-4 border-t border-gray-200 flex justify-end">
                                                        {pair.is_canonical ? (
                                                            <Badge className="bg-green-100 text-green-800">
                                                                Canonical
                                                            </Badge>
                                                        ) : (
                                                            <Badge className="bg-gray-100 text-gray-800">
                                                                Regular
                                                            </Badge>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 p-4 rounded-lg">
                                    <p>No merged QA pairs found. Merge some questions to see them here!</p>
                                </div>
                            )}
                        </div>
                    </TabsContent>
                </Tabs>
            </div>
        </MainLayout>
    );
}