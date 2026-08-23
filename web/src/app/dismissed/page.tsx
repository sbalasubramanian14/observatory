"use client";

import { StoryListPage } from "@/components/StoryListPage";
import { getDismissedIdsOrdered, unmarkDismissed } from "@/lib/personalization";

export default function DismissedPage() {
  return (
    <StoryListPage
      title="Dismissed"
      subtitle="Stories you've dismissed on this device, most recently dismissed first. Restoring one clears it from this list and lets it reappear in your feed again."
      getIdsOrdered={getDismissedIdsOrdered}
      removeId={unmarkDismissed}
      actionLabel="Restore"
      emptyTitle="Nothing dismissed"
      emptyBody="Dismissed stories are hidden from your feed, never deleted. Dismiss one by accident and it'll show up here, one tap away from coming back."
      secondaryLink={{ href: "/saved/", label: "View saved stories" }}
    />
  );
}
