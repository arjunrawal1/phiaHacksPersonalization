import { PipelineDashboard } from "./components/pipeline-dashboard";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-5 py-8">
      <header className="flex flex-col gap-2">
        <h1 className="text-4xl font-semibold tracking-tight">phia Personalization Pipeline</h1>
        <p className="text-sm text-muted-foreground">
          Live view of backend processing from camera roll upload to clothing labeling.
        </p>
        <p className="font-mono text-xs text-muted-foreground">Backend: {BACKEND_URL}</p>
      </header>

      <PipelineDashboard backendUrl={BACKEND_URL} />
    </main>
  );
}
