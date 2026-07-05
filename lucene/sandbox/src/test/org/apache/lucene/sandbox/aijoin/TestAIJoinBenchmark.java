/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.sandbox.aijoin;

import java.io.IOException;
import java.util.Locale;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.SearcherManager;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.JoinUtil;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;

/**
 * Wall-clock comparison of {@link JoinUtil} (the term-based variant, no global ordinals) against
 * {@link AIJoinQuery} on a synthetic M:1 join: ~100K children in a few segments point to ~10K
 * unique parents in a few segments through single valued sorted string docvalues. A small
 * child-side term filter selects ~10K children that join to ~1K parents; a second case adds a
 * parent-side filter on top of the join. Results are not asserted, only timed: each search repeats
 * {@link #PASSES} times and min/max/avg wall times are printed. The indices are never updated
 * after the build, and the AI join {@link SearcherManager} is created once and reused between
 * passes.
 */
public class TestAIJoinBenchmark extends LuceneTestCase {

  private static final String PARENT_ID = "parent_id";
  private static final String PARENT_ID_FK = "parent_id_FK";
  private static final String TAG = "tag";
  private static final String COLOR = "color";

  private static final String HOT = "hot";
  private static final String COLD = "cold";
  private static final String[] COLORS = {"red", "green", "blue"};

  private static final int NUM_PARENTS = 10_000;
  private static final int CHILDREN_PER_PARENT = 10;
  // every HOT_STRIDE'th parent is "hot" and all its children carry the hot tag: the child filter
  // TermQuery(tag:hot) matches NUM_PARENTS / HOT_STRIDE * CHILDREN_PER_PARENT ~ 10K children
  // joining to NUM_PARENTS / HOT_STRIDE ~ 1K parents, scattered over all segments
  private static final int HOT_STRIDE = 10;
  private static final int PARENT_SEGMENTS = 4;
  private static final int CHILD_SEGMENTS = 5;
  private static final int PASSES = 10;

  private interface SearchTask {
    TopDocs run() throws IOException;
  }

  public void testBenchmarkJoins() throws Exception {
    try (Directory parentsDir = newDirectory();
        Directory childrenDir = newDirectory()) {
      buildIndices(parentsDir, childrenDir);
      try (IndexReader parentsReader = DirectoryReader.open(parentsDir);
          IndexReader childrenReader = DirectoryReader.open(childrenDir)) {
        System.out.printf(
            Locale.ROOT,
            "parents: %d docs / %d segments, children: %d docs / %d segments%n",
            parentsReader.maxDoc(),
            parentsReader.leaves().size(),
            childrenReader.maxDoc(),
            childrenReader.leaves().size());
        // plain searchers without caching: AIJoinQuery is uncacheable anyway, so a cached
        // JoinUtil query would make the comparison lopsided
        IndexSearcher childrenSearcher = new IndexSearcher(childrenReader);
        childrenSearcher.setQueryCache(null);
        IndexSearcher parentsSearcher = new IndexSearcher(parentsReader);
        parentsSearcher.setQueryCache(null);

        Query childFilter = new TermQuery(new Term(TAG, HOT));
        Query parentFilter = new TermQuery(new Term(COLOR, COLORS[0]));

        // the auxiliary join index and its SearcherManager are built once and reused by all passes
        Directory joinDir = newDirectory();
        IndexWriter joinWriter =
            new IndexWriter(joinDir, newIndexWriterConfig(new MockAnalyzer(random())));
        long buildStart = System.nanoTime();
        AIJoinUtil.writeAJoinIndex(
            joinWriter, childrenReader, PARENT_ID_FK, parentsReader, PARENT_ID);
        System.out.printf(
            Locale.ROOT,
            "AI join index build: %.2fms%n",
            (System.nanoTime() - buildStart) / 1_000_000d);
        SearcherManager joinSearcherManager = new SearcherManager(joinWriter, null);

        // join query creation is inside the timed task on purpose: JoinUtil runs the from-side
        // selection eagerly in createJoinQuery, AIJoinQuery lazily in createWeight, so only
        // create+search is comparable
        bench(
            "JoinUtil",
            () ->
                parentsSearcher.search(
                    joinChildrenToParents(childFilter, childrenSearcher), 10));
        bench(
            "AIJoin",
            () ->
                parentsSearcher.search(
                    aiJoinChildrenToParents(
                        joinSearcherManager, childFilter, childrenSearcher, parentsReader),
                    10));
        bench(
            "JoinUtil + parent filter",
            () ->
                parentsSearcher.search(
                    filterParents(
                        joinChildrenToParents(childFilter, childrenSearcher), parentFilter),
                    10));
        bench(
            "AIJoin + parent filter",
            () ->
                parentsSearcher.search(
                    filterParents(
                        aiJoinChildrenToParents(
                            joinSearcherManager, childFilter, childrenSearcher, parentsReader),
                        parentFilter),
                    10));

        IOUtils.close(joinSearcherManager, joinWriter, joinDir);
      }
    }
  }

  private static Query joinChildrenToParents(Query childFilter, IndexSearcher childrenSearcher)
      throws IOException {
    // multipleValuesPerDocument=false picks the term-based join, no global ordinals involved
    return JoinUtil.createJoinQuery(
        PARENT_ID_FK, false, PARENT_ID, childFilter, childrenSearcher, ScoreMode.None);
  }

  private static Query aiJoinChildrenToParents(
      SearcherManager joinSearcherManager,
      Query childFilter,
      IndexSearcher childrenSearcher,
      IndexReader parentsReader) {
    return new AIJoinQuery(
        joinSearcherManager, PARENT_ID_FK, childFilter, childrenSearcher, parentsReader, PARENT_ID);
  }

  private static Query filterParents(Query joinQuery, Query parentFilter) {
    return new BooleanQuery.Builder()
        .add(joinQuery, BooleanClause.Occur.MUST)
        .add(parentFilter, BooleanClause.Occur.FILTER)
        .build();
  }

  private static void bench(String name, SearchTask task) throws IOException {
    long minNs = Long.MAX_VALUE;
    long maxNs = 0;
    long totalNs = 0;
    long hits = -1;
    for (int pass = 0; pass < PASSES; pass++) {
      long start = System.nanoTime();
      TopDocs topDocs = task.run();
      long elapsed = System.nanoTime() - start;
      minNs = Math.min(minNs, elapsed);
      maxNs = Math.max(maxNs, elapsed);
      totalNs += elapsed;
      hits = topDocs.totalHits.value();
    }
    System.out.printf(
        Locale.ROOT,
        "%-26s min=%8.2fms max=%8.2fms avg=%8.2fms hits=%d%n",
        name,
        minNs / 1_000_000d,
        maxNs / 1_000_000d,
        (double) totalNs / PASSES / 1_000_000d,
        hits);
  }

  /**
   * Writes the two sides with plain {@link IndexWriter}s: every child points to exactly one
   * parent, {@code parent_id} is unique on the parents side, both join fields are single valued
   * sorted string docvalues, and periodic commits under {@link NoMergePolicy} leave each side in a
   * few segments.
   */
  private void buildIndices(Directory parentsDir, Directory childrenDir) throws IOException {
    try (IndexWriter parentsWriter =
            new IndexWriter(
                parentsDir,
                newIndexWriterConfig(new MockAnalyzer(random()))
                    .setMergePolicy(NoMergePolicy.INSTANCE)
                    // pin the flush triggers: the randomized test config may pick tiny buffers
                    // and shatter the intended segment layout
                    .setMaxBufferedDocs(IndexWriter.MAX_DOCS)
                    .setRAMBufferSizeMB(256));
        IndexWriter childrenWriter =
            new IndexWriter(
                childrenDir,
                newIndexWriterConfig(new MockAnalyzer(random()))
                    .setMergePolicy(NoMergePolicy.INSTANCE)
                    // pin the flush triggers: the randomized test config may pick tiny buffers
                    // and shatter the intended segment layout
                    .setMaxBufferedDocs(IndexWriter.MAX_DOCS)
                    .setRAMBufferSizeMB(256))) {
      int parentsPerSegment = NUM_PARENTS / PARENT_SEGMENTS;
      int childrenPerSegment = NUM_PARENTS * CHILDREN_PER_PARENT / CHILD_SEGMENTS;
      int childSeq = 0;
      for (int p = 0; p < NUM_PARENTS; p++) {
        String parentId = "parent" + p;
        boolean hot = p % HOT_STRIDE == 0;

        Document parentDoc = new Document();
        parentDoc.add(new StringField(PARENT_ID, parentId, Field.Store.NO));
        parentDoc.add(new SortedDocValuesField(PARENT_ID, new BytesRef(parentId)));
        parentDoc.add(new StringField(COLOR, COLORS[p % COLORS.length], Field.Store.NO));
        parentsWriter.addDocument(parentDoc);
        if ((p + 1) % parentsPerSegment == 0) {
          parentsWriter.commit();
        }

        for (int c = 0; c < CHILDREN_PER_PARENT; c++) {
          Document childDoc = new Document();
          childDoc.add(new SortedDocValuesField(PARENT_ID_FK, new BytesRef(parentId)));
          childDoc.add(new StringField(TAG, hot ? HOT : COLD, Field.Store.NO));
          childrenWriter.addDocument(childDoc);
          if (++childSeq % childrenPerSegment == 0) {
            childrenWriter.commit();
          }
        }
      }
      parentsWriter.commit();
      childrenWriter.commit();
    }
  }
}
