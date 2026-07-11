package org.apache.lucene.sandbox.aijoin;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.MergePolicy.MergeContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.util.Bits;

final class AIJoinMergePolicy extends MergePolicy {
  @Override
  public MergePolicy.MergeSpecification findMerges(
      MergeTrigger mergeTrigger, SegmentInfos segmentInfos, MergeContext mergeContext)
      throws IOException {
    Set<SegmentCommitInfo> merging = mergeContext.getMergingSegments();
    MergeSpecification spec = null;
    for (SegmentCommitInfo info : segmentInfos) {
      if (merging.contains(info)) {
        continue;
      }
      Set<String> pairFieldNames = AIJoinUtil.pairFieldNames(AIJoinUtil.readFieldInfos(info));
      if (!pairFieldNames.isEmpty() && pendingPairRemovals.containsAll(pairFieldNames)) {//todo sweep pending removals as well
        if (spec == null) {
          spec = new MergeSpecification();
        }
        spec.add(new DropSegmentMerge(List.of(info)));
      }
    }
    return spec;
  }

  // counts merges that actually dropped a fully-dead segment; test-only observability, see
  // droppedSegmentCount()
  private final AtomicInteger droppedSegmentCount = new AtomicInteger();

  /** A merge over a single dead segment whose contents are reported as fully deleted, so {@link
   * IndexWriter} drops it instead of rewriting it -- see {@link #wrapForMerge}. Non-static so it
   * can report back to the outer policy's {@link #droppedSegmentCount}. */
  private final class DropSegmentMerge extends OneMerge {
    DropSegmentMerge(List<SegmentCommitInfo> segments) {
      super(segments);
    }

    @Override
    public CodecReader wrapForMerge(CodecReader reader) {
      return new FilterCodecReader(reader) {
        @Override
        public CacheHelper getCoreCacheHelper() {
          return reader.getCoreCacheHelper();
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
          return null; // we are altering live docs
        }

        @Override
        public Bits getLiveDocs() {
          return new Bits.MatchNoBits(reader.maxDoc());
        }

        @Override
        public int numDocs() {
          return 0;
        }
      };
    }

    @Override
    public void mergeFinished(boolean success, boolean segmentDropped) throws IOException {
      if (segmentDropped) {
        droppedSegmentCount.incrementAndGet();
      }
      super.mergeFinished(success, segmentDropped);
    }
  }

  /** Test-only: how many sidecar segments this policy has actually reaped so far. */
  int droppedSegmentCount() {
    return droppedSegmentCount.get();
  }

  /** Test-only: how many dead pair field names are currently queued for the next reap. */
  int pendingPairRemovalsCount() {
    return pendingPairRemovals.size();
  }

  @Override
  public MergePolicy.MergeSpecification findForcedMerges(
      SegmentInfos segmentInfos,
      int maxSegmentCount,
      Map<SegmentCommitInfo, Boolean> segmentsToMerge,
      MergeContext mergeContext)
      throws IOException {
    return null;
  }

  @Override
  public MergePolicy.MergeSpecification findForcedDeletesMerges(
      SegmentInfos segmentInfos, MergeContext mergeContext) throws IOException {
    return null;
  }

  // caps how many distinct (from-searcher, to-searcher) pairs we remember snapshots for; a
  // best-effort bound since this only anchors a heuristic reap, never correctness
  private static final int MAX_TRACKED_SEARCHER_PAIRS = 256;
  private final ConcurrentHashMap<Map.Entry<Object, Object>, Set<String>> lastNeededPairsBySearcherPair =
      new ConcurrentHashMap<>();
  private final ConcurrentLinkedQueue<Map.Entry<Object, Object>> trackedSearcherPairsOrder =
      new ConcurrentLinkedQueue<>();

  // pair field names seen in an earlier snapshot but missing from a later one for the same
  // (from-searcher, to-searcher) pair -- i.e. no longer needed -- queued here for findMerges to
  // reap; also size-capped, same reasoning
  private static final int MAX_PENDING_PAIR_REMOVALS = 4096;
  private final Set<String> pendingPairRemovals = ConcurrentHashMap.newKeySet();
  private final ConcurrentLinkedQueue<String> pendingPairRemovalsOrder = new ConcurrentLinkedQueue<>();

  protected void onCreateWeight(Set<String> neededPairs, IndexSearcher fromSearcher, IndexSearcher searcher) throws IOException {
    //TODO skip frequent invocations
    Object fromKey = AIJoinUtil.directoryKey(((DirectoryReader) fromSearcher.getIndexReader()).directory());
    Object toKey = AIJoinUtil.directoryKey(((DirectoryReader) searcher.getIndexReader()).directory());
    Map.Entry<Object, Object> searcherKey = Map.entry(fromKey, toKey);

    Set<String> currentSnapshot = Set.copyOf(neededPairs);
    Set<String> previousSnapshot =
        AIJoinIndex.putBounded(
            lastNeededPairsBySearcherPair,
            trackedSearcherPairsOrder,
            searcherKey,
            currentSnapshot,
            MAX_TRACKED_SEARCHER_PAIRS);
    if (previousSnapshot != null) {
      for (String pairFieldName : previousSnapshot) {
        if (!currentSnapshot.contains(pairFieldName)) {
          AIJoinMergePolicy.addBounded(
              pendingPairRemovals, pendingPairRemovalsOrder, pairFieldName, MAX_PENDING_PAIR_REMOVALS);
        }
      }
    }
  }

  /** Same eviction policy as {@link AIJoinIndex#putBounded}, for a plain set. */
  static <T> void addBounded(
      Set<T> set, ConcurrentLinkedQueue<T> insertionOrder, T value, int maxSize) {
    if (set.add(value)) {
      insertionOrder.add(value);
      while (set.size() > maxSize) {
        T oldest = insertionOrder.poll();
        if (oldest == null) {
          break;
        }
        set.remove(oldest);
      }
    }
  }
}
