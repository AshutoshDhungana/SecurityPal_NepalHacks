// API client for communicating with the FastAPI backend

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
    private async request<T>(
        endpoint: string,
        options: RequestInit = {}
    ): Promise<T | null> {
        try {
            const url = `${API_URL}${endpoint}`;

            // Add default headers
            const headers = {
                'Content-Type': 'application/json',
                ...(options.headers || {})
            };

            const response = await fetch(url, {
                ...options,
                headers
            });

            // Handle non-OK responses
            if (!response.ok) {
                console.error(`API error: ${response.status} ${response.statusText}`, await response.text());
                return null;
            }

            // Parse JSON response
            return await response.json() as T;
        } catch (error) {
            console.error('API request failed:', error);
            return null;
        }
    }

    // Get summary data
    async getSummary(productFilter?: string) {
        const params = productFilter ? `?product=${encodeURIComponent(productFilter)}` : '';
        return this.request<any>(`/summary${params}`);
    }

    // Get products list
    async getProducts() {
        return this.request<any[]>('/products');
    }

    // Get clusters
    async getClusters(params: {
        limit?: number;
        offset?: number;
        min_size?: number;
        product?: string;
        health_status?: string;
    } = {}) {
        const queryParams = new URLSearchParams();

        if (params.limit) queryParams.append('limit', params.limit.toString());
        if (params.offset) queryParams.append('offset', params.offset.toString());
        if (params.min_size) queryParams.append('min_size', params.min_size.toString());
        if (params.product) queryParams.append('product', params.product);
        if (params.health_status) queryParams.append('health_status', params.health_status);

        const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
        return this.request<any[]>(`/clusters${query}`);
    }

    // Get cluster entries
    async getClusterEntries(clusterId: string | number) {
        return this.request<any[]>(`/cluster/${clusterId}/entries`);
    }

    // Get similar pairs
    async getSimilarPairs(params: {
        min_similarity?: number;
        limit?: number;
        offset?: number;
        product?: string;
    } = {}) {
        const queryParams = new URLSearchParams();

        if (params.min_similarity) queryParams.append('min_similarity', params.min_similarity.toString());
        if (params.limit) queryParams.append('limit', params.limit.toString());
        if (params.offset) queryParams.append('offset', params.offset.toString());
        if (params.product) queryParams.append('product', params.product);

        const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
        return this.request<any[]>(`/similar-pairs${query}`);
    }

    // Get outdated entries
    async getOutdatedEntries(params: {
        min_days?: number;
        limit?: number;
        offset?: number;
        product?: string;
    } = {}) {
        const queryParams = new URLSearchParams();

        if (params.min_days) queryParams.append('min_days', params.min_days.toString());
        if (params.limit) queryParams.append('limit', params.limit.toString());
        if (params.offset) queryParams.append('offset', params.offset.toString());
        if (params.product) queryParams.append('product', params.product);

        const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
        return this.request<any[]>(`/outdated${query}`);
    }

    // Search similar questions
    async searchSimilarQuestions(query: string, product?: string, topK: number = 5) {
        const params: Record<string, string> = {
            query,
            top_k: topK.toString()
        };

        if (product) params.product = product;

        const queryParams = new URLSearchParams(params);
        return this.request<any[]>(`/search?${queryParams.toString()}`);
    }

    // Similarity search using embeddings
    async similaritySearch(searchData: {
        query: string;
        product_id?: string;
        category?: string;
        threshold?: number;
        top_k?: number;
    }) {
        return this.request<any>('/search/similarity', {
            method: 'POST',
            body: JSON.stringify(searchData)
        });
    }

    // Get embedding for text
    async getEmbedding(text: string) {
        return this.request<number[]>('/embedding', {
            method: 'POST',
            body: JSON.stringify({ text })
        });
    }

    // Merge QA pairs
    async mergeQAPairs(mergeRequest: any) {
        return this.request<any>('/merge-qa-pairs', {
            method: 'POST',
            body: JSON.stringify(mergeRequest)
        });
    }

    // Save merged QA pair
    async saveMergedPair(pairData: any) {
        return this.request<any>('/save-merged-pair', {
            method: 'POST',
            body: JSON.stringify(pairData)
        });
    }

    // Get merged QA pairs
    async getMergedQAPairs() {
        return this.request<any[]>('/merged-qa-pairs');
    }

    // Get content health trends
    async getContentHealthTrends(productFilter?: string, months: number = 6) {
        const params = new URLSearchParams();

        if (productFilter) params.append('product', productFilter);
        if (months) params.append('months', months.toString());

        const queryString = params.toString() ? `?${params.toString()}` : '';
        return this.request<any[]>(`/trends/content-health${queryString}`);
    }

    // Get pipeline steps
    async getPipelineSteps() {
        return this.request<any[]>('/pipeline/steps');
    }

    // Run pipeline
    async runPipeline(options: {
        product?: string | null;
        start_step: number;
        end_step: number;
        dry_run: boolean;
    }) {
        return this.request<any>('/pipeline/run', {
            method: 'POST',
            body: JSON.stringify(options)
        });
    }
}

const api = new ApiClient();
export default api;