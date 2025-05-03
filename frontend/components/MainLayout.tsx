'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard,
    Search,
    Network,
    ClipboardList,
    Clock,
    FileQuestion,
    Play,
    ChevronRight,
    ChevronLeft
} from 'lucide-react';

interface MainLayoutProps {
    children: React.ReactNode;
}

export default function MainLayout({ children }: MainLayoutProps) {
    const pathname = usePathname();
    const [collapsed, setCollapsed] = useState(false);

    const navigation = [
        { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
        { name: 'Cluster Explorer', href: '/clusters', icon: Network },
        { name: 'Similarity Search', href: '/similarity-search', icon: Search },
        { name: 'Outdated Content', href: '/outdated', icon: Clock },
        { name: 'Review Panel', href: '/review', icon: ClipboardList },
        { name: 'Pipeline Control', href: '/pipeline', icon: Play },
        { name: 'Upload Dataset', href: '/import', icon: FileQuestion }
    ];

    const isActive = (path: string) => {
        return pathname === path;
    };

    return (
        <div className="flex w-screen h-screen bg-gray-100">
            {/* Sidebar */}
            <div className={`bg-white shadow-lg transition-all duration-300 ${collapsed ? 'w-16' : 'w-64'}`}>
                <div className="flex flex-col h-full">
                    {/* Logo and header */}
                    <div className="flex items-center justify-between p-4 border-b">
                        {!collapsed && (
                            <div>
                                <h1 className="text-2xl font-bold text-blue-600">KLense</h1>
                                <p className="text-xs text-gray-500"></p>
                            </div>
                        )}
                        <button
                            onClick={() => setCollapsed(!collapsed)}
                            className="p-1 rounded-md hover:bg-gray-100"
                        >
                            {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
                        </button>
                    </div>

                    {/* Navigation links */}
                    <nav className="mt-6 flex-1 px-2">
                        <ul className="space-y-1">
                            {navigation.map((item) => (
                                <li key={item.name}>
                                    <Link
                                        href={item.href}
                                        className={`
                      flex items-center px-4 py-3 text-sm font-medium rounded-md
                      ${isActive(item.href)
                                                ? 'bg-blue-50 text-blue-600'
                                                : 'text-gray-700 hover:bg-gray-50 hover:text-blue-600'}
                      transition-colors
                    `}
                                    >
                                        <item.icon className={`${collapsed ? 'mx-auto' : 'mr-3'} h-5 w-5`} />
                                        {!collapsed && <span>{item.name}</span>}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </nav>

                    {/* Footer */}
                    <div className="p-4 border-t text-center text-xs text-gray-500">
                        {!collapsed && (
                            <>
                                <p>KLense v1.0</p>
                                <p className="mt-1"></p>
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* Main content */}
            <div className="flex-1 overflow-auto">
                <main className="py-6 px-4 sm:px-6 lg:px-8 w-full">
                    {children}
                </main>
            </div>
        </div>
    );
}