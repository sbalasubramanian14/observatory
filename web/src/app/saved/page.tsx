"use client";

import { StoryListPage } from "@/components/StoryListPage";
import {
  getSavedIdsOrdered,
  getSavedSnapshot,
  refreshSavedSnapshots,
  unmarkSaved,
} from "@/lib/personalization";

export default function SavedPage() {
  return (
    <StoryListPage
      title="Saved"
      subtitle="Stories you've saved on this device, most recently saved first. Saving records a signal only in this browser's local storage — it's per-device, there's no account, and it never leaves your browser. Saved stories are kept here even after they drop out of the feed's recent-news window."
      getIdsOrdered={getSavedIdsOrdered}
      removeId={unmarkSaved}
      // The feed carries only the last few days of news, so a story saved
      // last week is no longer in the bundle. Fall back to the copy taken
      // when it was saved — otherwise the bookmark button quietly stops
      // meaning anything after a few days.
      getFallback={getSavedSnapshot}
      onStoriesLoaded={refreshSavedSnapshots}
      actionLabel="Unsave"
      emptyTitle="Nothing saved yet"
      emptyBody="Tap the bookmark icon on any story to save it here for later. Saved stories live only in this browser, on this device — there's no account, no sync, and no server copy, so saving on your phone won't show up on your laptop."
      secondaryLink={{ href: "/dismissed/", label: "Review dismissed stories" }}
    />
  );
}
