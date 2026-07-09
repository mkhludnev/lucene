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

import java.io.Closeable;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.lucene.document.column.Column;
import org.apache.lucene.document.column.ColumnBatch;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NoMergeScheduler;
import org.apache.lucene.sandbox.aijoin.AIJoinUtil.DocMapping;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.SearcherManager;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.IOUtils;

/**
 * The auxiliary join index: a self-maintaining sidecar persisting per (from-segment, to-segment)
 * doc id mappings, so query-time joining reduces to bitset translation. It owns the sidecar's
 * {@link IndexWriter} and {@link SearcherManager}; pair columns are built lazily when an {@link
 * AIJoinQuery} first needs them, so users only {@link #open} (or {@link #inMemory()}) an instance
 * once, create queries with {@link #newJoinQuery} and search them with a bare to-side {@link
 * IndexSearcher}:
 *
 * <pre class="prettyprint">
 * AIJoinIndex joinIndex = AIJoinIndex.open(joinDir);   // once per process
 * Query q = joinIndex.newJoinQuery(fromField, fromQuery, fromSearcher, toField);
 * TopDocs hits = toSearcher.search(q, 10);
 * ...
 * joinIndex.close();                                   // app shutdown
 * </pre>
 *
 * <p>After either side reopens, the next query builds only the missing (from, to) segment pairs:
 * pair columns are addressed by both sides' persistent segment keys, which survive reopens. Pair
 * columns orphaned by merges are not reclaimed yet; see {@code README.md} in this package.
 */
public final class AIJoinIndex implements Closeable {

  private final Directory directory;
  private final IndexWriter writer;
  private final SearcherManager manager;

  /**
   * Dedups concurrent builders per pair field name: the thread that installs the future writes the
   * pair, others wait on it. Completed futures stay put so a builder that raced a not-yet-visible
   * refresh cannot write a duplicate pair column.
   */
  private final ConcurrentHashMap<String, CompletableFuture<Map.Entry<String, DocMapping>>> pairBuilds =
      new ConcurrentHashMap<>();

  /**
   * One (from-segment, to-segment) pair's ordinal-map column: its field name and the name of the
   * sidecar segment carrying it, so the column survives join reader refreshes.
   */
  record PairColumn(String pairFieldName, String joinSegmentName, int maxFromDoc, int [] toDocEdges) {}

  /** A pair's (from-segment, to-segment) leaf ordinals. */
  record SegmentsTuple(int fromLeafOrd, int toLeafOrd) {}

  private AIJoinIndex(Directory directory) throws IOException {
    this.directory = directory;
    this.writer =
        new IndexWriter(
            directory,
            new IndexWriterConfig()
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setMergeScheduler(NoMergeScheduler.INSTANCE));
    this.manager = new SearcherManager(writer, null);
  }

  /**
   * Opens a persistent auxiliary join index over the given directory, creating it if empty. Takes
   * ownership of the directory: {@link #close()} closes it.
   */
  public static AIJoinIndex open(Directory directory) throws IOException {
    return new AIJoinIndex(directory);
  }

  /** Opens a heap-resident auxiliary join index, rebuilt lazily from scratch every process run. */
  public static AIJoinIndex inMemory() throws IOException {
    return new AIJoinIndex(new ByteBuffersDirectory());
  }

  /**
   * Creates a query joining the docs matching {@code fromQuery} in {@code fromSearcher}'s index to
   * the index the returned query is executed against, through {@code fromField} = {@code toField}
   * term equality. Missing pair columns are built into this join index on first execution.
   */
  public Query newJoinQuery(
      String fromField, Query fromQuery, IndexSearcher fromSearcher, String toField) {
    return new AIJoinQuery(this, fromField, fromQuery, fromSearcher, toField);
  }

  IndexSearcher acquire() throws IOException {
    return manager.acquire();
  }

  void release(IndexSearcher searcher) throws IOException {
    manager.release(searcher);
  }

  /**
   * Builds and persists the given missing pair columns, keyed by pair field name to their
   * (from-segment, to-segment) leaf ordinals. Pairs concurrently built by another thread are
   * awaited, not rebuilt. On return the internal searcher manager is refreshed past every
   * requested pair.
   *
   * @return in memory data for just written segemts
   */
  Map<String, DocMapping> writeJoinSegments(
      Map<String, SegmentsTuple> missingPairs,
      IndexReader fromReader,
      String fromField,
      IndexReader toReader,
      String toField)
      throws IOException {
    Map<String, CompletableFuture<Map.Entry<String, DocMapping>>> owned = new LinkedHashMap<>();
    List<CompletableFuture<Map.Entry<String, DocMapping>>> awaited = new ArrayList<>();
    for (String pairFieldName : missingPairs.keySet()) {
      CompletableFuture<Map.Entry<String, DocMapping>> created = new CompletableFuture<>();
      CompletableFuture<Map.Entry<String, DocMapping>> existing = pairBuilds.putIfAbsent(pairFieldName, created);
      if (existing == null) {
        owned.put(pairFieldName, created);
      } else {
        awaited.add(existing);
      }
    }
    Map<String,DocMapping> loadedMappings = new LinkedHashMap<>();
    try {
      if (!owned.isEmpty()) {
        // all owned pairs go into a single batch: pair columns are addressed by from-side doc id,
        // so a batch must start at doc 0 of its sidecar segment, which writeBatch guarantees by
        // flushing one batch per commit
        List<Column> columns = new ArrayList<>();
        int batchNumDocs = 0;
        for (String pairFieldName : owned.keySet()) {
          SegmentsTuple position = missingPairs.get(pairFieldName);
          LeafReaderContext toContext = toReader.leaves().get(position.toLeafOrd());
          LeafReaderContext fromContext = fromReader.leaves().get(position.fromLeafOrd());
          long[] scratch =
              new long
                  [Math.toIntExact(
                      DocValues.getSortedSet(fromContext.reader(), fromField).getValueCount())];
          AIJoinUtil.DocMapping mapping =
              AIJoinUtil.computeDocMapping(fromContext, fromField, toContext, toField, scratch);
          columns.addAll( // what if there's no hits ???!! TODO needs to wite a thumbnail??
              AIJoinUtil.createJoinColumns(mapping, pairFieldName));
          batchNumDocs = Math.max(batchNumDocs, fromContext.reader().maxDoc());
          loadedMappings.put(pairFieldName, mapping);
        }
        writeBatch(batchNumDocs, columns);
        //TODO flush every single field to get single field segments
        for (Map.Entry<String, CompletableFuture<Map.Entry<String, DocMapping>>> entry :
            owned.entrySet()) {
          entry
              .getValue()
              .complete(
                  new AbstractMap.SimpleImmutableEntry<>(
                      entry.getKey(), loadedMappings.get(entry.getKey())));
        }
      }
    } catch (Throwable t) {
      // withdraw the claims so a later query can retry the build
      for (Map.Entry<String, CompletableFuture<Map.Entry<String, DocMapping>>> entry :
          owned.entrySet()) {
        pairBuilds.remove(entry.getKey(), entry.getValue());
        entry.getValue().completeExceptionally(t);
      }
      throw t;
    }
    Map<String, DocMapping> result = new LinkedHashMap<>(loadedMappings);
    for (CompletableFuture<Map.Entry<String, DocMapping>> future : awaited) {
      try {
        Map.Entry<String, DocMapping> entry = future.join();
        result.put(entry.getKey(), entry.getValue());
      } catch (CompletionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof IOException ioe) {
          throw ioe;
        }
        if (cause instanceof RuntimeException re) {
          throw re;
        }
        throw new IOException(cause);
      }
    }
    return result;
  }

  /**
   * Serializes sidecar writes: one batch per commit keeps every batch at doc 0 of its own segment,
   * preserving pair-column doc number == from-side doc id. Completing builders' futures after this
   * returns guarantees waiters observe the refreshed reader.
   */
  private synchronized void writeBatch(int batchNumDocs, List<Column> columns) throws IOException {
    writer.addBatch(
        new ColumnBatch() {
          @Override
          public int numDocs() {
            return batchNumDocs;
          }

          @Override
          public Iterable<Column> columns() {
            return columns;
          }
        });
    writer.commit();
    manager.maybeRefreshBlocking();
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(manager, writer, directory);
  }
}
