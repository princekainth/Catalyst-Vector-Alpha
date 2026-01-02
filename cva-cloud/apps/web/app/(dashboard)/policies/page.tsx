import SectionHeader from "@/components/section-header";
import { Card } from "@/components/ui/card";

export default function PoliciesPage() {
  return (
    <div className="space-y-8">
      <SectionHeader title="Policies">Configure auto-approval rules by cluster.</SectionHeader>
      <Card>
        <p className="text-sm text-white/70">
          Policy builder will appear here. For now, configure policies in your backend.
        </p>
      </Card>
    </div>
  );
}
