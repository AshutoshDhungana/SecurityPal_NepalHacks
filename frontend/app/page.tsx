'use client';

import { useState, useEffect } from 'react';
import MainLayout from '@/components/MainLayout';
import api from '@/lib/api';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { toast } from 'sonner';

export default function Dashboard() {
  const [summaryData, setSummaryData] = useState<any>(null);
  const [productData, setProductData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedProduct, setSelectedProduct] = useState<string | null>(null);

  useEffect(() => {
    async function fetchDashboardData() {
      try {
        setLoading(true);

        // Format product name with underscores if needed
        const formattedProduct = selectedProduct ? selectedProduct.replace(" ", "_") : null;
        const params = formattedProduct ? { product: formattedProduct } : undefined;

        // Fetch summary data
        const summary = await api.getSummary(formattedProduct || undefined);
        setSummaryData(summary);

        // Fetch product data
        const products = await api.getProducts();
        setProductData(products || []);

      } catch (error: any) {
        console.error('Error fetching dashboard data:', error);
        toast.error('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    }

    fetchDashboardData();
  }, [selectedProduct]);

  // Prepare health distribution data for the pie chart
  const getHealthData = () => {
    if (!summaryData || !summaryData.cluster_health) {
      // Default data if not available
      return [
        { name: 'Healthy', value: 33, color: '#4CAF50' },
        { name: 'Needs Review', value: 33, color: '#FF9800' },
        { name: 'Critical', value: 34, color: '#F44336' }
      ];
    }

    const healthData = summaryData.cluster_health;
    return [
      { name: 'Healthy', value: healthData.Healthy || 0, color: '#4CAF50' },
      { name: 'Needs Review', value: healthData['Needs Review'] || 0, color: '#FF9800' },
      { name: 'Critical', value: healthData.Critical || 0, color: '#F44336' }
    ];
  };

  return (
    <MainLayout>
      <div className="container mx-auto">
        <h1 className="text-3xl font-bold text-blue-600 mb-6">Dashboard</h1>

        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : summaryData ? (
          <>
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard
                title="Total Questions"
                value={summaryData.total_questions || 0}
                color="blue"
              />

              <MetricCard
                title="Total Clusters"
                value={summaryData.clusters?.total || 0}
                color="green"
              />

              <MetricCard
                title="Avg. Cluster Size"
                value={summaryData.avg_cluster_size ||
                  (summaryData.cluster_size_distribution?.mean || 0).toFixed(1)}
                color="orange"
                isDecimal
              />

              <MetricCard
                title="Health Score"
                value={summaryData.health_score ? (summaryData.health_score * 100).toFixed(1) :
                  (summaryData.questions?.canonical && summaryData.total_questions ?
                    ((summaryData.questions.canonical / summaryData.total_questions) * 100).toFixed(1) : 0)}
                color="purple"
                suffix="%"
                isDecimal
              />
            </div>

            {/* Cluster Health Distribution */}
            <div className="bg-white rounded-lg shadow-md p-6 mb-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Cluster Health Distribution</h2>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={getHealthData()}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="value"
                      nameKey="name"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {getHealthData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value} clusters`, 'Count']} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        ) : (
          <></>
        )}
      </div>
    </MainLayout>
  );
}

// Metric Card Component
interface MetricCardProps {
  title: string;
  value: number | string;
  color: string;
  suffix?: string;
  isDecimal?: boolean;
}

function MetricCard({ title, value, color, suffix = '', isDecimal = false }: MetricCardProps) {
  const colorVariants = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    orange: 'bg-orange-50 text-orange-600',
    purple: 'bg-purple-50 text-purple-600',
    red: 'bg-red-50 text-red-600',
  };

  const displayValue = typeof value === 'number' && !isDecimal
    ? value.toLocaleString()
    : value;

  return (
    <div className="bg-white rounded-lg shadow-md p-6 text-center hover:shadow-lg transition-shadow">
      <div className="text-2xl font-bold mb-2">
        <span className={`${colorVariants[color as keyof typeof colorVariants]}`}>
          {displayValue}{suffix}
        </span>
      </div>
      <div className="text-gray-500 font-medium">{title}</div>
    </div>
  );
}
