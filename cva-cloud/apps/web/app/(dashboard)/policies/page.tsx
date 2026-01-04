import SectionHeader from "@/components/section-header";
import PoliciesTableClient from "@/components/policies-table-client";

export default function PoliciesPage() {
  return (
    <div className="space-y-8">
      <SectionHeader title="Policies">Configure auto-approval rules by cluster.</SectionHeader>
      <PoliciesTableClient />
    </div>
  );
}
