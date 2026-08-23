"use client";

import { StoryListPage } from "@/components/StoryListPage";
import { getSavedIdsOrdered, unmarkSaved } from "@/lib/personalization";

export default function SavedPage() {
  return (
    <StoryListPage
      title="Saved"
      subtitle="Stories you've saved on this device, most recently saved first. Saving records a signal only in this browser's local storage — it's per-device, there's no account, and it never leaves your browser."
      getIdsOrdered={getSavedIdsOrdered}
      removeId={unmarkSaved}
      actionLabel="Unsave"
      emptyTitle="Nothing saved yet"
      emptyBody="Tap the bookmark icon on any story to save it here for later. Saved stories live only in this browser, on this device — there's no account, no sync, and no server copy, so saving on your phone won't show up on your laptop."
      secondaryLink={{ href: "/dismissed/", label: "Review dismissed stories" }}
    />
  );
}
