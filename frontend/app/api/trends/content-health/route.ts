import { NextRequest, NextResponse } from 'next/server';
import api from '@/lib/api';

export async function GET(request: NextRequest) {
    try {
        // Get data from the backend API
        const trendData = await api.getContentHealthTrends();

        // If the backend API fails, return mock data
        if (!trendData) {
            const mockData = [
                { month: "Jan", Outdated: 10, Similar: 15, Incomplete: 5, Healthy: 30 },
                { month: "Feb", Outdated: 9, Similar: 14, Incomplete: 5, Healthy: 32 },
                { month: "Mar", Outdated: 8, Similar: 12, Incomplete: 4, Healthy: 36 },
                { month: "Apr", Outdated: 9, Similar: 10, Incomplete: 4, Healthy: 37 },
                { month: "May", Outdated: 8, Similar: 8, Incomplete: 3, Healthy: 41 },
                { month: "Jun", Outdated: 6, Similar: 7, Incomplete: 2, Healthy: 45 },
            ];

            return NextResponse.json(mockData);
        }

        return NextResponse.json(trendData);
    } catch (error) {
        console.error('Error in content-health API route:', error);
        return NextResponse.json(
            { error: 'Failed to fetch content health trends' },
            { status: 500 }
        );
    }
}