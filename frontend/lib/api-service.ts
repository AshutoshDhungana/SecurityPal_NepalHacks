/**
 * API Service for interacting with the FastAPI backend
 * This connects to the api.py endpoints
 */

// Base URL for API endpoints
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Type definitions for the API responses
export interface ApiError {
  status: number;
  message: string;
  data?: any;
}

export interface SummaryData {
  product?: string;
  total_questions: number;
  clusters: {
    total: number;
    noise_points: number;
  };
  questions?: {
    canonical: number;
    redundant: number;
    archived?: number;
    active?: number;
    active_canonical?: number;
    archived_canonical?: number;
  };
  cluster_size_distribution?: {
    min: number;
    max: number;
    mean: number;
    median: number;
  };
  cluster_health: {
    Healthy: number;
    "Needs Review": number;
    Critical: number;
  };
  health_score: number;
}

export interface Product {
  product_id: string;
  product_name: string;
  description?: string;
}

export interface Cluster {
  cluster_id: string;
  size: number;
  health_status: string;
  similarity_score: number;
  product?: string;
  questions?: string[];
  canonical_questions?: string[];
  topics?: string[];
}

export interface QAEntry {
  question: string;
  answer: string;
  question_id?: string;
  is_canonical: boolean;
  last_updated?: string;
  cluster_id?: string;
  product?: string;
}

export interface SimilarPair {
  entry_id1: string;
  entry_id2: string;
  similarity_score: number;
}

export interface QAPair {
  id: string;
  question: string;
  answer: string;
  merged_at?: string;
  sources?: string[];
  is_canonical?: boolean;
  file_path?: string;
}

export interface MergeRequest {
  pair1: QAPair;
  pair2: QAPair;
  user_id: string;
  similarity_score: number;
  priority?: string;
}

export interface PipelineStep {
  name: string;
  file: string;
  args: string[];
}

export interface PipelineOptions {
  product?: string;
  start_step: number;
  end_step?: number;
  dry_run: boolean;
}

// Helper function to handle API errors
const handleApiError = async (response: Response): Promise<ApiError> => {
  const error: ApiError = {
    status: response.status,
    message: response.statusText || "Unknown error occurred",
  };
  
  try {
    const errorData = await response.json();
    error.message = errorData.detail || errorData.message || response.statusText;
  } catch {
    // If parsing fails, use the status text
  }
  
  return error;
};

// Generic fetch function with error handling
const fetchApi = async <T>(
  endpoint: string,
  options?: RequestInit,
  params?: Record<string, string | number | boolean | undefined>
): Promise<T> => {
  try {
    // Build URL with query parameters if provided
    let url = `${API_BASE_URL}${endpoint}`;
    
    if (params) {
      const queryParams = new URLSearchParams();
      
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          queryParams.append(key, String(value));
        }
      });
      
      const queryString = queryParams.toString();
      if (queryString) {
        url = `${url}?${queryString}`;
      }
    }
    
    // Make the request
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        ...(options?.headers || {}),
      },
    });
    
    // Handle error responses
    if (!response.ok) {
      const error = await handleApiError(response);
      throw error;
    }
    
    // Parse and return the response
    return await response.json() as T;
  } catch (error) {
    console.error(`API Error on ${endpoint}:`, error);
    throw error;
  }
};

// API service methods
export const apiService = {
  // Dashboard summary
  async getSummary(product?: string): Promise<SummaryData> {
    return fetchApi<SummaryData>(
      "/summary",
      undefined,
      product ? { product } : undefined
    );
  },
  
  // Get products list
  async getProducts(): Promise<Product[]> {
    return fetchApi<Product[]>("/products");
  },
  
  // Get clusters with optional filtering
  async getClusters(params: {
    limit?: number;
    offset?: number;
    min_size?: number;
    product?: string;
    health_status?: string;
  } = {}): Promise<Cluster[]> {
    return fetchApi<Cluster[]>("/clusters", undefined, params);
  },
  
  // Get detailed information about a specific cluster
  async getClusterDetails(clusterId: string): Promise<Cluster> {
    return fetchApi<Cluster>(`/clusters/${clusterId}`);
  },
  
  // Get all entries in a specific cluster
  async getClusterEntries(clusterId: string): Promise<QAEntry[]> {
    return fetchApi<QAEntry[]>(`/cluster/${clusterId}/entries`);
  },
  
  // Get outdated entries
  async getOutdatedEntries(params: {
    min_days?: number;
    limit?: number;
    offset?: number;
    product?: string;
  } = {}): Promise<QAEntry[]> {
    return fetchApi<QAEntry[]>("/outdated", undefined, params);
  },
  
  // Get pairs of entries with high similarity
  async getSimilarPairs(params: {
    min_similarity?: number;
    limit?: number;
    offset?: number;
    product?: string;
  } = {}): Promise<SimilarPair[]> {
    return fetchApi<SimilarPair[]>("/similar-pairs", undefined, params);
  },
  
  // Get available pipeline steps
  async getPipelineSteps(): Promise<PipelineStep[]> {
    return fetchApi<PipelineStep[]>("/pipeline/steps");
  },
  
  // Run the pipeline with specific options
  async runPipeline(options: PipelineOptions): Promise<{ message: string; options: PipelineOptions }> {
    return fetchApi<{ message: string; options: PipelineOptions }>(
      "/pipeline/run",
      {
        method: "POST",
        body: JSON.stringify(options),
      }
    );
  },
  
  // Get merged QA pairs
  async getMergedQAPairs(): Promise<QAPair[]> {
    return fetchApi<QAPair[]>("/merged-qa-pairs");
  },
  
  // Merge two QA pairs
  async mergeQAPairs(mergeRequest: MergeRequest): Promise<QAPair> {
    return fetchApi<QAPair>(
      "/merge-qa-pairs",
      {
        method: "POST",
        body: JSON.stringify(mergeRequest),
      }
    );
  },
  
  // Save a merged QA pair
  async saveMergedPair(pair: QAPair): Promise<{ success: boolean; file_path: string; merged_pair: QAPair }> {
    return fetchApi<{ success: boolean; file_path: string; merged_pair: QAPair }>(
      "/save-merged-pair",
      {
        method: "POST",
        body: JSON.stringify(pair),
      }
    );
  },
  
  // Search for similar questions
  async searchSimilarQuestions(query: string, params: {
    product?: string;
    top_k?: number;
    min_similarity?: number;
  } = {}): Promise<{ question: string; answer: string; similarity: number }[]> {
    // The existing api.py doesn't seem to have a direct endpoint for similarity search
    // So we'll simulate this with a POST request and assume it exists or will be added
    return fetchApi<{ question: string; answer: string; similarity: number }[]>(
      "/search",
      {
        method: "POST",
        body: JSON.stringify({ query, ...params })
      }
    );
  },
  
  // Get cluster visualizations
  async getClusterVisualization(params: {
    product?: string;
    method?: string;
  } = {}): Promise<string> {
    // This returns HTML content directly
    const response = await fetch(
      `${API_BASE_URL}/visualizations/clusters${params ? 
        `?${new URLSearchParams(
          Object.entries(params)
            .filter(([_, v]) => v !== undefined)
            .map(([k, v]) => [k, String(v)])
        )}` : ''}`
    );
    
    if (!response.ok) {
      const error = await handleApiError(response);
      throw error;
    }
    
    return await response.text();
  },
  
  // Get health dashboard visualization
  async getHealthDashboard(): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/visualizations/health-dashboard`);
    
    if (!response.ok) {
      const error = await handleApiError(response);
      throw error;
    }
    
    return await response.text();
  },
  
  // Upload a file
  async uploadFile(file: File): Promise<any> {
    const formData = new FormData();
    formData.append("file", file);
    
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });
    
    if (!response.ok) {
      const error = await handleApiError(response);
      throw error;
    }
    
    return await response.json();
  },
};

export default apiService;
