export interface User {
  id: string;
  email: string;
  displayName: string | null;
  createdAt: string;
}

export interface Workspace {
  id: string;
  name: string;
  slug: string;
  createdAt: string;
  memberCount: number;
  documentCount: number;
}

export interface Document {
  id: string;
  filename: string;
  fileType: string;
  fileSize: number;
  chunkCount: number;
  status: string;
  uploadedAt: string;
  processedAt: string | null;
}

export interface ChunkInfo {
  chunkId: string;
  content: string;
  relevanceScore: number;
  documentName: string;
}

export interface ClaimVerdict {
  claim: string;
  verdict: "SUPPORTED" | "PARTIALLY_SUPPORTED" | "NOT_SUPPORTED";
  evidence: string | null;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  chunksUsed: ChunkInfo[];
  claims: ClaimVerdict[];
  retries: number;
  latencyMs: number;
}

export interface HistoryItem {
  id: string;
  queryText: string;
  responseText: string | null;
  confidenceScore: number | null;
  chunksUsed: number | null;
  createdAt: string;
}

export interface Subscription {
  plan: "starter" | "pro" | "enterprise";
  status: string;
  currentPeriodEnd: string | null;
}

export interface Plan {
  tier: "starter" | "pro" | "enterprise";
  name: string;
  price: number;
  queriesPerDay: number;
  documentsLimit: number;
  storageMb: number;
  features: string[];
}
