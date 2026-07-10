package org.apache.lucene.sandbox.aijoin;

import java.io.IOException;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.sandbox.aijoin.AIJoinIndex.PairColumn;
import org.apache.lucene.sandbox.aijoin.AIJoinIndex.SegmentsTuple;
import org.apache.lucene.sandbox.aijoin.AIJoinQuery.JoinSegmentReference;
import org.apache.lucene.sandbox.aijoin.AIJoinUtil.DocEdges;
import org.apache.lucene.sandbox.aijoin.AIJoinUtil.DocMapping;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;

/**
 * TODO prune by "to" range if it's under slice searching
 * TODO pass raw cacheless searcher
*/
class ToLeafJoinContext {
  final LeafReaderContext toContext;
  final Query fromQuery;
  final IndexSearcher cachedFromSearcher;
  private IndexSearcher lastSeenJoinSearcher;
  final String fromField;
  final String toField;
  final private AIJoinIndex joinIndex;

// TODO all of these might be final since they are set in the constructor
  private int firstToDoc = DocIdSetIterator.NO_MORE_DOCS;
  private int lastToDoc = -1;
  private long matchedToDocsCount=0;
  private FixedBitSet toApproximation=null;
  /** ordered by from-cost descending */
  private final List<JoinTask> joinCells = new ArrayList<>();
  // secondary indices over joinCells, kept in sync by addJoinTask/removeJoinCell: every cell is
  // reachable both by its from-segment ordinal (dense, so a plain array) and by its pair field
  // name (sparse across the full from-segment space, so a map)
  private final JoinTask[] joinCellsByFromSegOrd;
  private final Map<String, JoinTask> joinCellsByPairFieldName = new HashMap<>();
  private IndexReader toReader;

  /**
   * How a resolved {@link JoinTask}'s edges and to-doc values are read: either a join-index
   * -persisted {@link PairColumn}, addressed through the cell's (possibly refreshed) {@link
   * JoinTask#joinSegmentRef}, or a {@link DocMapping} built on demand and held entirely in
   * memory. Replaces a pair of nullable {@code JoinTask} fields so a cell is resolved as exactly
   * one of the two, never both, never neither.
   */
  sealed interface FromToRelation extends DocEdges {
    /** it's like PARENT_FK_DOCNUM */
    SortedNumericDocValues toDocsByFromDocsDV(JoinTask owner, IndexSearcher lastSeenJoinSearcher)
        throws IOException;
  }

  record ByJoinIndexSegment(PairColumn pairColumn) implements FromToRelation {
    @Override
    public int[] fromDocEdges() {
      return pairColumn.fromDocEdges();
    }

    @Override
    public int[] toDocEdges() {
      return pairColumn.toDocEdges();
    }

    @Override
    public SortedNumericDocValues toDocsByFromDocsDV(JoinTask owner, IndexSearcher lastSeenJoinSearcher)
        throws IOException {
      // validate the join segment is still present
      LeafReaderContext joinContext =
          lastSeenJoinSearcher.getLeafContexts().get(owner.joinSegmentRef.joinSegmentLeafOrd());
      assert AIJoinUtil.segmentName(joinContext).equals(owner.joinSegmentRef.joinSegmentName());
      return joinContext
          .reader()
          .getSortedNumericDocValues(
              AIJoinUtil.TO_DOC_VAL_BY_FROM_DOCNUM + pairColumn.pairFieldName());
    }
  }

  record ByArray(DocMapping docMapping) implements FromToRelation {
    @Override
    public int[] fromDocEdges() {
      return docMapping.fromDocEdges();
    }

    @Override
    public int[] toDocEdges() {
      return docMapping.toDocEdges();
    }

    @Override
    public SortedNumericDocValues toDocsByFromDocsDV(JoinTask owner, IndexSearcher lastSeenJoinSearcher) {
      // this join segment was not found in the join index, so it was built on demand and is
      // now available via docMapping directly, with no searcher involved
      return docMapping.toDocByFromDoc();
    }
  }

  /** represents a cell in the join matrix bounded to from and to segments */
  class JoinTask implements DocEdges {
    final String pairFieldName;
    final DocIdSetIterator fromSegmentDocIdIter;
    JoinSegmentReference joinSegmentRef;
    final SegmentsTuple segmentsFromTo;
    private FromToRelation resolved;

    JoinTask(
        String pairFieldName,
        SegmentsTuple segmentsFromTo,
        DocIdSetIterator fromSegmentDocIdIter) {
      this.pairFieldName = pairFieldName;
      this.segmentsFromTo = segmentsFromTo;
      this.fromSegmentDocIdIter = fromSegmentDocIdIter;
      assert fromSegmentDocIdIter.docID() != DocIdSetIterator.NO_MORE_DOCS;
      assert fromSegmentDocIdIter.docID() >= 0;
    }

    /** Resolves this cell, once, to a join-index-persisted pair column. */
    void resolveColumn(PairColumn pairColumn) {
      assert resolved == null : "already resolved: " + resolved;
      resolved = new ByJoinIndexSegment(pairColumn);
    }

    /** Resolves this cell, once, to an in-memory doc mapping built on demand. */
    void resolveMapping(DocMapping docMapping) {
      assert resolved == null : "already resolved: " + resolved;
      resolved = new ByArray(docMapping);
    }

    boolean isResolved() {
      return resolved != null;
    }

    SortedNumericDocValues toDocsByFromDocsDV() throws IOException {
      assert resolved != null : "not resolved yet: " + this;
      return resolved.toDocsByFromDocsDV(this, ToLeafJoinContext.this.lastSeenJoinSearcher);
    }

    @Override
    public int[] fromDocEdges() {
      return resolved.fromDocEdges();
    }

    @Override
    public int[] toDocEdges() {
      return resolved.toDocEdges();
    }
  }

  /**
   * Appends {@code cell} to {@link #joinCells} and registers it in both indices.
   */
  private JoinTask addJoinTask(JoinTask cell) {
    joinCells.add(cell);
    joinCellsByFromSegOrd[cell.segmentsFromTo.fromLeafOrd()] = cell;
    joinCellsByPairFieldName.put(cell.pairFieldName, cell);
    return cell;
  }

  /**
   * Drops {@code cell} from {@link #joinCells} and both indices, so a pair found to no longer
   * contribute disappears from every view of it at once.
   */
  private void removeJoinCell(JoinTask cell) {
    joinCells.remove(cell);
    joinCellsByFromSegOrd[cell.segmentsFromTo.fromLeafOrd()] = null;
    joinCellsByPairFieldName.remove(cell.pairFieldName);
  }

  /** Looks up a cell by from-segment ordinal, or {@code null} if none is registered there. */
  JoinTask cellByFromSegOrd(int fromSegOrd) {
    return joinCellsByFromSegOrd[fromSegOrd];
  }

  /** Looks up a cell by pair field name, or {@code null} if none is registered under that name. */
  JoinTask cellByPairFieldName(String pairFieldName) {
    return joinCellsByPairFieldName.get(pairFieldName);
  }

  /**
   * A lazy view of every cell's weight-age {@link JoinSegmentReference} as a (pairFieldName, segment)
   * entry, skipping cells without one -- equivalent to iterating a {@code Map<String,
   * JoinSegment>} of just the weight-age segments, but read off {@link JoinTask#joinSegmentRef}
   * instead of keeping a separate map around.
   */
  private Iterable<Map.Entry<String, JoinSegmentReference>> weightAgeJoinSegmentEntries() {
    return () ->
        joinCellsByPairFieldName.entrySet().stream()
            .filter(e -> e.getValue().joinSegmentRef != null)
            .<Map.Entry<String, JoinSegmentReference>>map(
                e -> Map.entry(e.getKey(), e.getValue().joinSegmentRef))
            .iterator();
  }

  record TaskRefreshResult(
     Set<Map.Entry<JoinTask,LeafReaderContext>> joinSegements,
    Set<Map.Entry<JoinTask,DocMapping>> justWritten
  ){}
  /**
   *
   * @param weightAgeJoinSegmentsReadOnly the join segments cached at {@link AIJoinQuery#createWeight} time DON'T MODIFY ME!!!
   * @param weightAgeJoinSearcher the join searcher cached at {@link AIJoinQuery#createWeight} time
   * @param scorerSupplierAgeJoinSearcher the join searcher used at {@link AIJoinWeight#scorerSupplier} time
   */

  ToLeafJoinContext( LeafReaderContext toContext, String fromField,
     Query fromQuery, IndexSearcher cachedFromSearcher, String toField,
      IndexReader toReader, Map<String, JoinSegmentReference> weightAgeJoinSegmentsReadOnly,
       IndexSearcher weightAgeJoinSearcher, IndexSearcher scorerSupplierAgeJoinSearcher,
       AIJoinIndex joinIndex) throws IOException {
    this.toContext = toContext;
    this.fromField = fromField;
    this.fromQuery = fromQuery;
    this.cachedFromSearcher = cachedFromSearcher;
    this.toField = toField;
    this.toReader = toReader;

    this.joinIndex = joinIndex;
    this.joinCellsByFromSegOrd = new JoinTask[this.cachedFromSearcher.getLeafContexts().size()];

    // 1. check from scorers
    for (JoinTask newJoinTask : createFromItersTasks()) {
      this.addJoinTask(newJoinTask);
      // 2. set old segment refernces
      JoinSegmentReference oldReference = weightAgeJoinSegmentsReadOnly.get(newJoinTask.pairFieldName);
      if (oldReference != null) {
        newJoinTask.joinSegmentRef = oldReference;
      }
    }

    this.lastSeenJoinSearcher = weightAgeJoinSearcher;//set old searcher, it correspondts to weightAgeJoinSegmentsReadOnly
    TaskRefreshResult refreshedAndNew = refreshJoinTasksReferences(scorerSupplierAgeJoinSearcher);
    for(Entry<JoinTask, LeafReaderContext> entry : refreshedAndNew.joinSegements){
      JoinTask cell = entry.getKey();
      LeafReaderContext joinLeaf = entry.getValue(); //got it from searcher leafs by refOrd
      assert AIJoinUtil.segmentName(joinLeaf).equals(cell.joinSegmentRef.joinSegmentName());
      assert joinLeaf.ord == cell.joinSegmentRef.joinSegmentLeafOrd();
      cell.resolveColumn( // ok. this one may be ready for search.
          new PairColumn(
              cell.pairFieldName,
              cell.joinSegmentRef.joinSegmentName(), //should be in sync
              cell.joinSegmentRef.joinSegmentLeafOrd(),
              ToLeafJoinContext.loadEdges(joinLeaf, AIJoinUtil.FROM_EDGES_PREFIX + cell.pairFieldName),
              ToLeafJoinContext.loadEdges(joinLeaf, AIJoinUtil.TO_EDGES_PREFIX + cell.pairFieldName),
              // TODO use it for ordefing join segment iteration, desc
              ToLeafJoinContext.loadEdges(joinLeaf, AIJoinUtil.TO_COUNT_PREFIX + cell.pairFieldName)[0]
            ));
    }
    for (Entry<JoinTask, DocMapping> entry : refreshedAndNew.justWritten) {
      JoinTask cell = entry.getKey();
      cell.resolveMapping(entry.getValue());
    }
    // edges are  loaded
    for (JoinTask cell : List.copyOf(joinCells)){ // hell. it might remove task from the list. that's sad. I have to copy it.
      assert cell.isResolved();
      // a little bit tricky. It assumes that column is join-index backed or just written and array-backed,
      advanceAtMinFromEdge(cell);
    }
    // now let's read each cell's pairColumn, then docMapping, and build "to" side bitset of approximation
    // first pass: union the contributing pairs' to-doc ranges; every possible match in this
    // to segment falls into [minToDoc, maxToDoc]
    for (JoinTask cell : joinCells) {
      DocEdges docEdges = cell;
      firstToDoc = Math.min(firstToDoc, docEdges.toDocEdges()[0]);
      lastToDoc = Math.max(lastToDoc, docEdges.toDocEdges()[1]);
      matchedToDocsCount += docEdges.toDocEdges()[1] - docEdges.toDocEdges()[0] + 1;
      if (toApproximation == null) {
        toApproximation = new FixedBitSet(toContext.reader().maxDoc());
      }
      toApproximation.set(docEdges.toDocEdges()[0], docEdges.toDocEdges()[1] + 1);
    }
  }

  private TaskRefreshResult refreshJoinTasksReferences(IndexSearcher newJoinIndexSearcher) throws IOException {
    Set<Map.Entry<JoinTask,LeafReaderContext>> joinSegements = new LinkedHashSet<>();
    Set<Map.Entry<JoinTask,DocMapping>> justWritten = new LinkedHashSet<>();

    Set<JoinTask> refeshReference = new LinkedHashSet<>();
    Set<JoinTask> loadReference = new LinkedHashSet<>();
    Map<String,JoinTask> needIndex = new LinkedHashMap<>();
    for (JoinTask task : joinCells) {
      JoinSegmentReference oldReference = task.joinSegmentRef;
      if (oldReference != null) {
        task.joinSegmentRef = oldReference;
        if (newJoinIndexSearcher==this.lastSeenJoinSearcher) {
          loadReference.add(task);
        } else {
          refeshReference.add(task);
        }
      } else {
        needIndex.put(task.pairFieldName, task);
      }
    }
    // referesh old refs, pass 1
    List<LeafReaderContext> newLeaves = newJoinIndexSearcher.getLeafContexts();
    for(Iterator<JoinTask> iter = refeshReference.iterator(); iter.hasNext(); ) {
      JoinTask task = iter.next();
      // check segment name by ord, if true, put to load, remove from here
      LeafReaderContext byOrd =
          task.joinSegmentRef.joinSegmentLeafOrd() < newLeaves.size()
              ? newLeaves.get(task.joinSegmentRef.joinSegmentLeafOrd())
              : null;
      if (byOrd != null &&
        AIJoinUtil.segmentName(byOrd).equals(task.joinSegmentRef.joinSegmentName())) {
        loadReference.add(task);
        iter.remove();
      }
    }
    // pass 2
    if (!refeshReference.isEmpty()) {
      Map<String, JoinTask> byOldJoinSegName  = new HashMap<>();
      for (JoinTask task : refeshReference) {
        byOldJoinSegName.put(task.joinSegmentRef.joinSegmentName(), task);
      }
      for (LeafReaderContext joinLeaf : newLeaves) {
        String segName = AIJoinUtil.segmentName(joinLeaf);
        JoinTask task = byOldJoinSegName.get(segName);
        if (task != null) { //TODO we got a segment right here
          task.joinSegmentRef =
              new JoinSegmentReference(
                  task.joinSegmentRef.pairFieldName(), segName, joinLeaf.ord);
          loadReference.add(task);
          refeshReference.remove(task);
        }
      }
    }
    // pass 3
    if (!refeshReference.isEmpty()) {
      Map<String, JoinTask> byPairFieldName  = new HashMap<>();
      for (JoinTask task : refeshReference) {
        byPairFieldName.put(task.joinSegmentRef.pairFieldName(), task);
      }
      // loop join segments search for fields
      Map<String, JoinSegmentReference> joinSegmentsByPairFieldName = AIJoinQuery.extractExistingJoinColumns(lastSeenJoinSearcher, byPairFieldName::containsKey);
      // if found move to load set
      for (JoinTask task :byPairFieldName.values()){
        JoinSegmentReference found = joinSegmentsByPairFieldName.get(task.pairFieldName);
        if (found != null) {
          task.joinSegmentRef = found;
          loadReference.add(task);
          refeshReference.remove(task);
        }
      }
    }
    if (!refeshReference.isEmpty()) { //TODO presumabily we can go to index it
      throw new IllegalStateException("unable to refresh segment refs " +
          refeshReference + " at " + lastSeenJoinSearcher);
    }
    // load edges for regulars
    for (JoinTask cell : loadReference) {
      //String pairFieldName = cell.pairFieldName;
      LeafReaderContext joinFeafSeg = lastSeenJoinSearcher.getLeafContexts().get(cell.joinSegmentRef.joinSegmentLeafOrd());
      assert AIJoinUtil.segmentName(joinFeafSeg).equals(cell.joinSegmentRef.joinSegmentName());
      joinSegements.add(new SimpleEntry<>(cell, joinFeafSeg));
    }
    // index for unlucked
    if (!needIndex.isEmpty()) {

      Map<String, SegmentsTuple> missingPairs = new HashMap<>();
      for (JoinTask cell : needIndex.values()) {
        missingPairs.put(cell.pairFieldName, cell.segmentsFromTo);
      }
      Map<String, DocMapping> written = this.joinIndex.writeJoinSegments(
        Collections.unmodifiableMap(missingPairs),
       cachedFromSearcher.getIndexReader()
        , this.fromField, this.toReader, this.toField);
      assert written.keySet().containsAll(missingPairs.keySet());
      assert missingPairs.keySet().containsAll(written.keySet());
      for (Map.Entry<String, DocMapping> entry : written.entrySet()) {//TODO optimize
        JoinTask cell = needIndex.get(entry.getKey());
        justWritten.add(new SimpleEntry<>(cell, entry.getValue()));
      }
    }
    this.lastSeenJoinSearcher = newJoinIndexSearcher;//set old searcher, it correspondts to weightAgeJoinSegmentsReadOnly
    return new TaskRefreshResult(joinSegements, justWritten);
  }

  /**
   * Positions {@code cell}'s from-iterator behind {@code docEdges}'s first from-doc, or drops
   * {@code cell} via {@link #removeJoinCell} when the iterator can no longer reach any doc the
   * pair maps -- either because the pair maps nothing ({-1, -1} sentinel), the iterator already
   * moved past the pair's last from-doc, or it exhausts before reaching the pair's first one (the
   * iterator only moves forward, so none of these are recoverable). Returns whether the cell
   * survives.
   */
  private boolean advanceAtMinFromEdge(JoinTask cell) throws IOException {
    int[] fromDocEdges = cell.fromDocEdges();
    int minFromDoc = fromDocEdges[0];
    int maxFromDoc = fromDocEdges[1];
    DocIdSetIterator fromSegemtIter = cell.fromSegmentDocIdIter;
    if (minFromDoc < 0) {
      // {-1, -1} sentinel: this pair maps no from doc to any to doc at all
      removeJoinCell(cell);
      return false;
    }
    if (maxFromDoc >=0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
      && fromSegemtIter.docID() > maxFromDoc) {
      // from iter is already past the last from doc this pair maps, so it cannot contribute
      removeJoinCell(cell); // no more matches in this join segment, so the pair cannot contribute
      return false;
    }
    if (minFromDoc >= 0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
      &&  fromSegemtIter.docID() < minFromDoc) {
      int firstMatch = fromSegemtIter.advance(minFromDoc);
      if (firstMatch == DocIdSetIterator.NO_MORE_DOCS || firstMatch > maxFromDoc) {
        /// wow from iter exhausted, no match in this join segment, so the pair cannot contribute
        // thus we need to return them from request `
        removeJoinCell(cell); // no more matches in this join segment, so the pair cannot contribute
        return false;
      }// else from iter is advanced behind the first from match , good
    }
    return true;
  }

  /**
   * every to segment call for prepositioned from seg iters
   * populates a {@link JoinTask} per contributing from segment, its iterator PREPOSITIONED to the
   * first matching doc
   * @return tasks are orfered by descending from-side match count, so the first task is the one with the most matches
   * @throws IOException
   */
  private List<JoinTask> createFromItersTasks() throws IOException {
    // TODO peek in underneath searcher cache
    Weight cachedFromWeight = this.cachedFromSearcher.createWeight(this.fromQuery,
        ScoreMode.COMPLETE_NO_SCORES, 1);

    List<LeafReaderContext> leaves = new ArrayList<>(this.cachedFromSearcher.getLeafContexts());
    Collections.shuffle(leaves, ThreadLocalRandom.current());

    List<JoinTask> tasks = new ArrayList<>();
    // the from-side scorer's cost() at task-build time, i.e. before it was drained looking for the
    // first match; auxiliary to this method only, just to sort tasks by descending match volume
    Map<JoinTask, Long> fromMatchCostByTask = new IdentityHashMap<>();
    for (LeafReaderContext fromContext :
            leaves) {
      //
      ScorerSupplier fromSupplier = cachedFromWeight.scorerSupplier(fromContext);
      if (fromSupplier == null) {
        continue; // no from-side matches in this segment
      }
      long fromMatchCost = fromSupplier.cost();
      Scorer fromScorer = fromSupplier.get(Long.MAX_VALUE);
      DocIdSetIterator matchedFromDocs = fromScorer.iterator();
      Bits liveDocs = fromContext.reader().getLiveDocs();
      if (liveDocs != null) {
        // the cached weight's scorer doesn't filter deletions itself, and a from doc deleted
        // since the pair columns were built (e.g. by an update) must not resolve to a match
        matchedFromDocs =
            new FilteredDocIdSetIterator(matchedFromDocs) {
              @Override
              protected boolean match(int doc) {
                return liveDocs.get(doc);
              }
            };
      }
      if (matchedFromDocs.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
        // name every contributing (from, to) pair column; pair field names are unique across pairs
        String pairFieldName =
            AIJoinUtil.pairFieldName(fromContext, this.fromField, toContext, this.toField);
        JoinTask task = new JoinTask(pairFieldName,
          new SegmentsTuple(fromContext.ord, toContext.ord),
           matchedFromDocs);
        tasks.add(task);
        fromMatchCostByTask.put(task, fromMatchCost);
      }
    }
    // process the from segments with the most matches first
    tasks.sort(Comparator.comparingLong(fromMatchCostByTask::get).reversed());
    return tasks;
  }

  /** Reads a pair's persisted values, all stored on doc 0 of the column.
   * TODO move it somewhere
  */
  static int[] loadEdges(LeafReaderContext joinContext, String edgesFieldName)
      throws IOException {
    SortedNumericDocValues edgesDV =
        joinContext.reader().getSortedNumericDocValues(edgesFieldName);
    assert edgesDV != null : "expected edges column to be present: " + edgesFieldName;
    int zeroDoc = edgesDV.nextDoc();
    assert zeroDoc == 0
        : "expected edges column to be fully materialized, but got doc " + zeroDoc;
    int[] values = new int[edgesDV.docValueCount()];
    for (int i = 0; i < values.length; i++) {
      values[i] = (int) edgesDV.nextValue();
    }
    return values;
  }

  public ScorerSupplier scorerSupplier(ScoreMode scoreMode, float boost) {
    if (toApproximation == null || matchedToDocsCount == 0) {
      return null; // no matches in this to segment
    }
    return new ScorerSupplier() {
      @Override
      public Scorer get(long leadCost) throws IOException {
        DocIdSetIterator approximation =
            new BitSetIterator(toApproximation, matchedToDocsCount

            );
        TwoPhaseIterator twoPhase =
            new TwoPhaseIterator(approximation) {
              boolean pruned = false;

              @Override
              public boolean matches() throws IOException {
                if (!pruned) {
                  FixedBitSet matchedToDocs;
                  int shift;
                  // prune the approximation to the resolved matches: drop the
                  // remaining range bits and or the matches back in at their
                  // absolute positions, so the approximation stops visiting
                  // non-matching docs
                  shift = approximation.docID();
                  // TODO the this is: it's enough to just confirm the match the return the control.
                  // TODO there, should be a refined bitset
                  // TODO when we need to refine an approx, we go throug join segments,
                  // TODO and drop them into refined bitset until we confirm the match
                  // TODO the order of iteration is: to side doc freq, which we neeed to persist and read then
                  // TODO once we drop a join segment to refined bitset we exclude it from iterations
                  // TODO also, just boundary check the following matches checks
                  // TODO the following matches() checks, at first confirm with refied bitset,
                  // TODO if it's false procede with to bitset dumping into refined bitset
                  matchedToDocs = refineToMatches( shift);
                  toApproximation.clear(shift, lastToDoc + 1);
                  FixedBitSet.orRange(matchedToDocs, 0, toApproximation, shift, lastToDoc - shift + 1);
                  pruned = true;
                  return toApproximation.get(approximation.docID());
                }
                // the bitset spans [shift, lastToDoc] shifted to zero
                // return matchedToDocs.get(approximation.docID() - shift);
                assert /*return*/ toApproximation.get(approximation.docID()); // always true ??
                return true;
              }

              @Override
              public float matchCost() {
                return matchedToDocsCount;
              }
            };
        return new ConstantScoreScorer(boost, scoreMode, twoPhase);
      }

      @Override
      public long cost() {
        return lastToDoc - firstToDoc + 1;
      }
    };
  }

  /**
   * TODO this refines false positeve approximation ,
   * but it can iteratively refine false-negative docset,
   * this let us giveup looping join tasks
  */
  private FixedBitSet refineToMatches(int shift)
      throws IOException {
    FixedBitSet matchedToDocs = new FixedBitSet(lastToDoc - shift + 1);
    IndexSearcher freshSearcher = this.joinIndex.acquire();
    try {

      refreshJoinTasksReferences(freshSearcher);
      assert this.lastSeenJoinSearcher == freshSearcher;
      for (JoinTask cell : joinCells) {
        SortedNumericDocValues toDocsByFromDoc = cell.toDocsByFromDocsDV();
        dumpToMatches(shift, matchedToDocs, cell, cell, toDocsByFromDoc);
      }
    } finally {//TODO release before looping remaining cells separately
      this.joinIndex.release(freshSearcher);
    }
    return matchedToDocs;
  }

  static private void dumpToMatches(int shift, FixedBitSet matchedToDocs, JoinTask cell, DocEdges docEdges, SortedNumericDocValues toDocsByFromDoc) throws IOException {
    DocIdSetIterator matchedFromDocs = cell.fromSegmentDocIdIter;
    for (int fromDoc = matchedFromDocs.docID(); // prepositioned to the first match
        fromDoc != DocIdSetIterator.NO_MORE_DOCS && fromDoc <= docEdges.fromDocEdges()[1]; fromDoc = matchedFromDocs.nextDoc()) {
      if (toDocsByFromDoc.advanceExact(fromDoc)) {
        for (int i = 0; i < toDocsByFromDoc.docValueCount(); i++) {
          int toDocMatch = (int) toDocsByFromDoc.nextValue();
          assert toDocMatch <= docEdges.toDocEdges()[1] && toDocMatch >= docEdges.toDocEdges()[0] : "to doc " + toDocMatch + " above edges union max " + Arrays.toString(docEdges.toDocEdges());
          // shift is wherever the approximation iterator first landed, which need not be
          // the global firstToDoc (e.g. under a boolean conjunction); a match below shift
          // is unreachable -- the iterator only moves forward -- so it's dropped rather
          // than written at a negative offset
          if (toDocMatch >= shift) {
            matchedToDocs.set(toDocMatch - shift);
          }
        }
      }
    }
  }
}
