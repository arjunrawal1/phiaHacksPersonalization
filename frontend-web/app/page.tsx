import { PipelineDashboard } from "./components/pipeline-dashboard";
import { PersonalizationPanel } from "./components/personalization-panel";
import { StylingLab } from "./components/styling-lab";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export default function Home() {
  return (
    <main className="phia-shell mx-auto flex w-full max-w-[1420px] flex-col gap-6 px-4 pb-10 pt-6 sm:px-6 lg:px-8">
      <header className="phia-hero overflow-hidden p-5 sm:p-7">
        <div>
          <div>
            <p className="phia-brandword text-3xl italic leading-none text-foreground sm:text-[2.2rem]">phia</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-[-0.025em] text-foreground sm:text-[2.3rem]">
              Personalization Pipeline
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground sm:text-[0.95rem]">
              Live view of backend processing from camera roll upload to clothing labeling.
            </p>
          </div>
        </div>
      </header>

      <p className="px-1 text-sm text-muted-foreground">
        Behind the scenes debugging dashboard for the same face and closet flow used in mobile.
      </p>

      <PipelineDashboard backendUrl={BACKEND_URL} />
      <PersonalizationPanel backendUrl={BACKEND_URL} />

      <StylingLab backendUrl={BACKEND_URL} />
    </main>
  );
}
