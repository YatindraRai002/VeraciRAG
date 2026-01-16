"use client";

import { useState, useEffect } from "react";
import { billingApi } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import { CheckIcon } from "@heroicons/react/24/outline";

interface Plan {
  id: string;
  name: string;
  price: number;
  features: string[];
  query_limit: number;
  document_limit: number;
}

export default function BillingPage() {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [currentPlan, setCurrentPlan] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [plansRes, subRes] = await Promise.all([
          billingApi.getPlans(),
          billingApi.getSubscription(),
        ]);
        setPlans(plansRes.data);
        setCurrentPlan(subRes.data?.plan_id || "starter");
      } catch (error) {
        console.error("Failed to fetch billing data:", error);
        // Default plans if API fails
        setPlans([
          {
            id: "starter",
            name: "Starter",
            price: 0,
            features: ["100 queries/month", "5 documents", "Basic support"],
            query_limit: 100,
            document_limit: 5,
          },
          {
            id: "pro",
            name: "Pro",
            price: 29,
            features: ["1,000 queries/month", "50 documents", "Priority support", "API access"],
            query_limit: 1000,
            document_limit: 50,
          },
          {
            id: "enterprise",
            name: "Enterprise",
            price: 99,
            features: ["Unlimited queries", "Unlimited documents", "24/7 support", "Custom integrations"],
            query_limit: -1,
            document_limit: -1,
          },
        ]);
        setCurrentPlan("starter");
      }
    };
    fetchData();
  }, []);

  const handleUpgrade = async (planId: string) => {
    setLoading(true);
    try {
      const response = await billingApi.createCheckout(planId);
      if (response.data.url) {
        window.location.href = response.data.url;
      }
    } catch (error) {
      console.error("Failed to create checkout:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Billing</h1>
        <p className="mt-1 text-sm text-gray-500">
          Manage your subscription and billing
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {plans.map((plan) => {
          const isCurrent = currentPlan === plan.id;
          return (
            <Card
              key={plan.id}
              className={isCurrent ? "ring-2 ring-blue-500" : ""}
            >
              <CardHeader>
                <CardTitle>{plan.name}</CardTitle>
                <CardDescription>
                  <span className="text-3xl font-bold text-gray-900">
                    ${plan.price}
                  </span>
                  <span className="text-gray-500">/month</span>
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 mb-6">
                  {plan.features.map((feature, i) => (
                    <li key={i} className="flex items-center text-sm text-gray-600">
                      <CheckIcon className="h-4 w-4 text-green-500 mr-2 flex-shrink-0" />
                      {feature}
                    </li>
                  ))}
                </ul>
                {isCurrent ? (
                  <Button variant="secondary" className="w-full" disabled>
                    Current Plan
                  </Button>
                ) : (
                  <Button
                    className="w-full"
                    onClick={() => handleUpgrade(plan.id)}
                    loading={loading}
                    variant={plan.price === 0 ? "outline" : "primary"}
                  >
                    {plan.price === 0 ? "Downgrade" : "Upgrade"}
                  </Button>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Usage This Month</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <p className="text-sm font-medium text-gray-500">Queries Used</p>
              <p className="mt-1 text-2xl font-semibold text-gray-900">
                0 / {plans.find((p) => p.id === currentPlan)?.query_limit || 100}
              </p>
              <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: "0%" }} />
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500">Documents</p>
              <p className="mt-1 text-2xl font-semibold text-gray-900">
                0 / {plans.find((p) => p.id === currentPlan)?.document_limit || 5}
              </p>
              <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full" style={{ width: "0%" }} />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
