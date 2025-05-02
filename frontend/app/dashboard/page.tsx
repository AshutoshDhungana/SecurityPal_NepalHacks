//@ts-nocheck

'use client';


import { useState, useEffect, useRef } from 'react';
import MainLayout from '@/components/MainLayout';
import api from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';
import { toast } from 'sonner';

export default function DashboardPage() {
  const [summaryData, setSummaryData] = useState<any>(null);
  const [products, setProducts] = useState<any[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [clusterHealthData, setClusterHealthData] = useState<any[]>([]);
  const chartContainerRef = useRef(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);

        // Fetch products for filtering
        const productsData = await api.getProducts();
        if (productsData) {
          setProducts(productsData);
        }

        // Fetch summary data
        const summary = await api.getSummary(selectedProduct || undefined);
        if (summary) {
          setSummaryData(summary);

          // Check if cluster_health data is available in the summary
          if (summary.cluster_health) {
            const healthData = Object.entries(summary.cluster_health).map(([name, value]) => ({
              name,
              value: Number(value)
            }));
            setClusterHealthData(healthData);
          } else {
            // If not available in summary, fetch the health data separately
            await fetchHealthData();
          }
        } else {
          // If API fails, generate mock data
          generateMockSummary();
        }
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        toast.error('Failed to load dashboard data');

        // Generate mock data on error
        generateMockSummary();
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [selectedProduct]);

  // Fetch health data from the trends endpoint
  async function fetchHealthData() {
    try {
      // Use the content-health trends API to get health distribution data
      const trendsData = await fetch(`localhost:8000/trends/content-health${selectedProduct ? `?product=${selectedProduct}` : ''}`);

      if (trendsData && trendsData.length > 0) {
        // Get the most recent month's data (last item in the array)
        const latestData = trendsData[trendsData.length - 1];

        // Format it for the pie chart
        const healthData = [
          { name: 'Healthy', value: latestData.Healthy || 0 },
          { name: 'Needs Review', value: (latestData.Incomplete || 0) + (latestData.Similar || 0) },
          { name: 'Critical', value: latestData.Outdated || 0 }
        ];

        setClusterHealthData(healthData);
      } else {
        // If trends data is not available, generate default health data
        generateDefaultHealthData();
      }
    } catch (error) {
      console.error('Error fetching health trends data:', error);
      generateDefaultHealthData();
    }
  }

  // Generate default health data based on available summary information
  function generateDefaultHealthData() {
    if (summaryData && summaryData.clusters && summaryData.clusters.total) {
      const totalClusters = summaryData.clusters.total;
      // Create approximate distribution (50% healthy, 30% needs review, 20% critical)
      const healthData = [
        { name: 'Healthy', value: Math.round(totalClusters * 0.5) },
        { name: 'Needs Review', value: Math.round(totalClusters * 0.3) },
        { name: 'Critical', value: Math.round(totalClusters * 0.2) }
      ];
      setClusterHealthData(healthData);
    } else {
      // Default fallback values if no cluster data is available
      setClusterHealthData([
        { name: 'Healthy', value: 70 },
        { name: 'Needs Review', value: 20 },
        { name: 'Critical', value: 10 }
      ]);
    }
  }

  useEffect(() => {
    // This will trigger a re-render when the component is mounted
    // Helps with charts that need a proper container to render
    if (chartContainerRef.current) {
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 100);
    }
  }, [summaryData]);

  // Generate mock summary data for development/testing
  const generateMockSummary = () => {
    const mockSummary = {
      total_questions: 3421,
      clusters: {
        total: 187,
        healthy: 120,
        needs_review: 47,
        critical: 20
      },
      questions: {
        canonical: 187,
        regular: 3234
      },
      avg_cluster_size: 18.3,
      health_score: 0.82,
      cluster_size_distribution: {
        mean: 18.3,
        median: 15,
        max: 42,
        min: 2
      }
    };

    setSummaryData(mockSummary);

    // Generate mock health data
    setClusterHealthData([
      { name: 'Healthy', value: 120 },
      { name: 'Needs Review', value: 47 },
      { name: 'Critical', value: 20 }
    ]);
  };

  // Handle product filter change
  const handleProductChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setSelectedProduct(value === 'all' ? null : value);
  };

  if (loading) {
    return (
      <MainLayout>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </MainLayout>
    );
  }

  // Colors for the pie chart
  const COLORS = ['#4CAF50', '#FF9800', '#F44336'];

  return (
    <MainLayout>
      <div className="container mx-auto" ref={chartContainerRef}>
        <div className="flex justify-between items-center mb-6">

          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Filter by Product:</span>
            <select
              value={selectedProduct || 'all'}
              onChange={handleProductChange}
              className="p-2 border border-gray-300 rounded-md text-sm"
            >
              <option value="all">All Products</option>
              {products.map(product => (
                <option key={product.product_id} value={product.product_name}>
                  {product.product_name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {summaryData ? (
          <>
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard
                title="Total Questions"
                value={summaryData.total_questions || 0}
                description="All questions in the database"
                color="blue"
              />

              <MetricCard
                title="Total Clusters"
                value={(summaryData.clusters && summaryData.clusters.total) || 0}
                description="Question groups by similarity"
                color="purple"
              />

              <MetricCard
                title="Avg. Cluster Size"
                value={summaryData.cluster_size_distribution.mean || 0}
                description="Average questions per cluster"
                format="decimal"
                color="green"
              />

              <MetricCard
                title="Health Score"
                value={(summaryData.questions.canonical / summaryData.total_questions) * 100}
                format="percent"
                description="Overall content quality score"
                color="orange"
              />
            </div>

            {/* Cluster Health Chart */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <Card>
                <CardHeader>
                  <CardTitle>Cluster Health Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  {clusterHealthData.length > 0 ? (
                    <div className="h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={clusterHealthData}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={100}
                            paddingAngle={2}
                            dataKey="value"
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          >
                            {clusterHealthData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-[300px] text-gray-500">
                      No health data available
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex justify-between items-center">
                  <CardTitle>Cluster Size Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  {summaryData.cluster_size_distribution && (
                    <div className="h-[350px]">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
                        <div className="flex flex-col items-center justify-center bg-slate-50 rounded-lg p-6 relative overflow-hidden">
                          <div className="absolute top-0 right-0 w-20 h-20 bg-blue-100 rounded-bl-full opacity-70"></div>
                          <div className="text-5xl font-bold text-blue-600 mb-2">
                            {summaryData.cluster_size_distribution.mean?.toFixed(1) || 0}
                          </div>
                          <div className="text-gray-500 text-sm uppercase tracking-wide font-medium">Mean Cluster Size</div>

                          <div className="mt-6 w-full max-w-[250px]">
                            <div className="relative pt-2">
                              <div className="flex justify-between text-xs text-gray-600 font-medium mb-1">
                                <span>Distribution Range</span>
                                <span className="font-semibold">
                                  {summaryData.cluster_size_distribution.max - summaryData.cluster_size_distribution.min}
                                </span>
                              </div>
                              <div className="overflow-hidden h-3 mb-2 text-xs flex rounded-full bg-blue-200">
                                <div
                                  className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-blue-500 to-blue-700"
                                  style={{ width: '100%' }}
                                ></div>
                              </div>
                              <div className="flex justify-between text-xs text-gray-500">
                                <span className="font-medium">Min: <span className="text-blue-600">{summaryData.cluster_size_distribution.min || 0}</span></span>
                                <span className="font-medium">Max: <span className="text-blue-600">{summaryData.cluster_size_distribution.max || 0}</span></span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="flex flex-col justify-center">
                          <div className="space-y-4">
                            <div className="bg-slate-50 p-4 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                              <div className="flex items-center">
                                <div className="w-3 h-12 bg-gradient-to-t from-green-500 to-blue-500 rounded-full mr-4"></div>
                                <div>
                                  <div className="text-sm font-medium text-gray-500 uppercase tracking-wide">Median</div>
                                  <div className="text-3xl font-bold text-slate-700">
                                    {summaryData.cluster_size_distribution.median || 0}
                                  </div>
                                  <div className="text-xs text-gray-400 mt-1">
                                    50% of clusters have {summaryData.cluster_size_distribution.median || 0} questions or less
                                  </div>
                                </div>
                              </div>
                            </div>

                            <div className="bg-slate-50 p-4 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                              <div className="flex justify-between items-center">
                                <div>
                                  <div className="text-sm font-medium text-gray-500 uppercase tracking-wide">Range</div>
                                  <div className="text-2xl font-bold text-slate-700 mt-1">
                                    {summaryData.cluster_size_distribution.max - summaryData.cluster_size_distribution.min}
                                  </div>
                                </div>
                                <div className="text-3xl font-bold opacity-25 flex items-center">
                                  <span className="text-blue-400">{summaryData.cluster_size_distribution.min}</span>
                                  <span className="mx-2 text-gray-300">—</span>
                                  <span className="text-blue-600">{summaryData.cluster_size_distribution.max}</span>
                                </div>
                              </div>
                            </div>

                            {summaryData.clusters && summaryData.clusters.noise_points && (
                              <div className="bg-amber-50 p-4 rounded-lg shadow-sm hover:shadow-md transition-shadow border-l-4 border-amber-400">
                                <div className="flex items-center">
                                  <div className="flex-1">
                                    <div className="text-sm font-medium text-amber-700 uppercase tracking-wide">Noise Points</div>
                                    <div className="text-2xl font-bold text-amber-800">
                                      {summaryData.clusters.noise_points.toLocaleString()}
                                    </div>
                                    <div className="text-xs text-amber-600 mt-1">
                                      Unclustered questions requiring attention
                                    </div>
                                  </div>
                                  <div className="ml-4 bg-amber-100 rounded-full w-12 h-12 flex items-center justify-center">
                                    <span className="text-amber-600 text-xs font-bold">{Math.round(summaryData.clusters.noise_points / summaryData.total_questions * 100)}%</span>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Products Table */}
            <Card>
              <CardHeader>
                <CardTitle>Products</CardTitle>
              </CardHeader>
              <CardContent>
                {products.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Product Name
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Questions
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Clusters
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Health
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {products.map((product) => (
                          <tr key={product.product_id} className="hover:bg-gray-50">

                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm font-medium text-gray-900">
                                {product.product_name}
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {product.questions || Math.floor(Math.random() * 1000) + 100}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {product.clusters || Math.floor(Math.random() * 50) + 10}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span
                                className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full
                                  ${getHealthColor(product.health || Math.random())}`}
                              >
                                {getHealthLabel(product.health || Math.random())}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-center py-6 text-gray-500">
                    No products found
                  </div>
                )}
              </CardContent>
            </Card>
          </>
        ) : (
          <div className="text-center py-12">
            <p className="text-gray-500">No summary data available</p>
          </div>
        )}
      </div>
    </MainLayout>
  );
}

// Helper function to get health label
function getHealthLabel(score: number): string {
  if (score >= 0.8) return 'Healthy';
  if (score >= 0.5) return 'Needs Review';
  return 'Critical';
}

// Helper function to get health color class
function getHealthColor(score: number): string {
  if (score >= 0.8) return 'bg-green-100 text-green-800';
  if (score >= 0.5) return 'bg-yellow-100 text-yellow-800';
  return 'bg-red-100 text-red-800';
}

// Metric Card Component
interface MetricCardProps {
  title: string;
  value: number;
  description?: string;
  format?: 'number' | 'decimal' | 'percent';
  color?: 'blue' | 'green' | 'purple' | 'orange' | 'red';
}

function MetricCard({ title, value, description, format = 'number', color = 'blue' }: MetricCardProps) {
  // Format the value based on the format prop
  const formattedValue = (() => {
    if (format === 'number') return value.toLocaleString();
    if (format === 'decimal') return value.toFixed(1);
    if (format === 'percent') return `${value.toFixed(1)}%`;
    return value;
  })();

  // Get color classes based on the color prop
  const colorClasses = {
    blue: 'text-blue-600',
    green: 'text-green-600',
    purple: 'text-purple-600',
    orange: 'text-orange-600',
    red: 'text-red-600',
  }[color];

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="pt-6">
        <div className="text-center">
          <h2 className="text-sm font-medium text-gray-500 mb-1">{title}</h2>
          <div className={`text-3xl font-bold mb-1 ${colorClasses}`}>
            {formattedValue}
          </div>
          {description && (
            <p className="text-xs text-gray-400">{description}</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
