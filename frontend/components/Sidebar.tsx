'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    Home,
    Layers,
    Search,
    GitCompare,
    Clock,
    ClipboardCheck,
    Terminal
} from 'lucide-react';

interface SidebarProps {
    products: any[];
    selectedProduct: string | null;
    onProductChange: (product: string | null) => void;
}

export default function Sidebar({ products, selectedProduct, onProductChange }: SidebarProps) {
    const pathname = usePathname();

    const navigation = [
        { name: 'Dashboard', href: '/', icon: Home },
        { name: 'Cluster Explorer', href: '/clusters', icon: Layers },
        { name: 'Similarity Search', href: '/similarity-search', icon: Search },
        { name: 'Similarity Analysis', href: '/similarity-analysis', icon: GitCompare },
        { name: 'Outdated Content', href: '/outdated', icon: Clock },
        { name: 'Review Panel', href: '/review', icon: ClipboardCheck },
        { name: 'Pipeline Control', href: '/pipeline', icon: Terminal }
    ];

    return (
        <div className="flex flex-col h-full bg-white border-r border-gray-200">
            <div className="flex items-center justify-center p-4 border-b">
                <div className="flex items-center gap-2">
                    <div className="bg-blue-600 p-2 rounded-lg">
                        <span className="text-white text-xl font-bold">?</span>
                    </div>
                    <h1 className="text-xl font-semibold text-gray-800">QnA Management</h1>
                </div>
            </div>

            <div className="flex flex-col flex-grow px-4 py-6">
                <h3 className="font-medium text-gray-700 mb-4">Navigation</h3>
                <nav className="flex-1 space-y-1">
                    {navigation.map((item) => (
                        <Link
                            key={item.name}
                            href={item.href}
                            className={`
                flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors
                ${pathname === item.href
                                    ? 'bg-blue-50 text-blue-600'
                                    : 'text-gray-700 hover:bg-gray-100'}
              `}
                        >
                            <item.icon
                                className={`mr-3 h-5 w-5 ${pathname === item.href ? 'text-blue-600' : 'text-gray-400'}`}
                            />
                            {item.name}
                        </Link>
                    ))}
                </nav>

                <div className="mt-6 pt-6 border-t border-gray-200">
                    <h3 className="font-medium text-gray-700 mb-4">Filters</h3>
                    <div>
                        <label htmlFor="product-select" className="block text-sm font-medium text-gray-700 mb-1">
                            Product
                        </label>
                        <select
                            id="product-select"
                            className="block w-full p-2 text-sm border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                            value={selectedProduct || ''}
                            onChange={(e) => onProductChange(e.target.value === '' ? null : e.target.value)}
                        >
                            <option value="">All Products</option>
                            {products.map((product) => (
                                <option key={product.product_name} value={product.product_name}>
                                    {product.product_name}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            <div className="p-4 border-t border-gray-200 text-center text-xs text-gray-500">
                <p>Dashboard v1.0</p>
                <p>© 2023 SecurityPal</p>
            </div>
        </div>
    );
}