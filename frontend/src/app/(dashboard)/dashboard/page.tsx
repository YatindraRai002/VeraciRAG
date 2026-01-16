"use client";

import { useWorkspace } from "@/hooks/useWorkspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  ClockIcon,
  ChartBarIcon,
} from "@heroicons/react/24/outline";

export default function DashboardPage() {
  const { currentWorkspace } = useWorkspace();

  const stats = [
    {
      name: "Documents",
      value: currentWorkspace?.document_count || 0,
      icon: DocumentTextIcon,
      color: "bg-blue-500",
    },
    {
      name: "Queries Today",
      value: 0,
      icon: ChatBubbleLeftRightIcon,
      color: "bg-green-500",
    },
    {
      name: "Total Queries",
      value: currentWorkspace?.query_count || 0,
      icon: ClockIcon,
      color: "bg-purple-500",
    },
    {
      name: "Avg Confidence",
      value: "85%",
      icon: ChartBarIcon,
      color: "bg-orange-500",
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Welcome to your VeraciRAG workspace
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <Card key={stat.name}>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className={`p-3 rounded-lg ${stat.color}`}>
                  <stat.icon className="h-6 w-6 text-white" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">{stat.name}</p>
                  <p className="text-2xl font-semibold text-gray-900">{stat.value}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <a
                href="/documents"
                className="flex items-center p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <DocumentTextIcon className="h-5 w-5 text-blue-600 mr-3" />
                <span className="text-sm font-medium text-gray-700">Upload Documents</span>
              </a>
              <a
                href="/chat"
                className="flex items-center p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <ChatBubbleLeftRightIcon className="h-5 w-5 text-green-600 mr-3" />
                <span className="text-sm font-medium text-gray-700">Start a Chat</span>
              </a>
              <a
                href="/history"
                className="flex items-center p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <ClockIcon className="h-5 w-5 text-purple-600 mr-3" />
                <span className="text-sm font-medium text-gray-700">View History</span>
              </a>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Getting Started</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-start">
                <div className="flex-shrink-0 h-6 w-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-sm font-medium">
                  1
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">Upload your documents</p>
                  <p className="text-sm text-gray-500">
                    Add PDF, TXT, or MD files to your workspace
                  </p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="flex-shrink-0 h-6 w-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-sm font-medium">
                  2
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">Ask questions</p>
                  <p className="text-sm text-gray-500">
                    Query your documents using natural language
                  </p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="flex-shrink-0 h-6 w-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-sm font-medium">
                  3
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">Get verified answers</p>
                  <p className="text-sm text-gray-500">
                    Our multi-agent system fact-checks every response
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
