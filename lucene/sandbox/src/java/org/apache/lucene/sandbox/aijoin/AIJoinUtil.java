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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.column.Column;
import org.apache.lucene.document.column.ColumnBatch;
import org.apache.lucene.document.column.LongColumn;
import org.apache.lucene.document.column.LongColumn.NumericKind;
import org.apache.lucene.document.column.LongTupleCursor;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NoMergeScheduler;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.StringHelper;

/**
 * Utilities for building and addressing the auxiliary AI join index: for every (from-segment,
 * to-segment) pair it holds a SORTED_NUMERIC column named {@link #pairFieldName}, whose doc number
 * is the from-side doc id and whose value is the to-side doc id whose {@code toField} term equals
 * the from doc's {@code fromField} term.
 */
final class AIJoinUtil {


  private static final FieldType toDocsFieldType = new FieldType();

  static {
    toDocsFieldType.setDocValuesType(DocValuesType.SORTED_NUMERIC);
    toDocsFieldType.freeze();
  }


  private AIJoinUtil() {}

  /**
   * Writes the doc-map columns for all (from-segment, to-segment) pairs into {@code writer} as a
   * single column-oriented batch and commits. Every pair column shares the batch's doc space
   * {@code [0, maxFromDoc)}, so a from-side doc id is the doc number in any pair column.
   */
  static void writeAJoinIndex(
      IndexWriter writer,
      IndexReader fromReader,
      String fromField,
      IndexReader toReader,
      String toField)
      throws IOException {
    // first pass: pair columns are addressed by from-side doc id, so the batch doc space must
    // cover the largest from-segment maxDoc; the merge scratch needs the largest ordinal space
    long maxFromValueCount = 0;
    int maxFromDoc = 0;
    for (LeafReaderContext fromContext : fromReader.leaves()) {
      maxFromValueCount =
          Math.max(
              maxFromValueCount,
              DocValues.getSortedSet(fromContext.reader(), fromField).getValueCount());
      maxFromDoc = Math.max(maxFromDoc, fromContext.reader().maxDoc());
    }
    final int batchNumDocs = maxFromDoc;
    if (batchNumDocs > 0 && maxFromValueCount > 0) {
      long[] scratch = new long[Math.toIntExact(maxFromValueCount)];
      // materialize all pair columns up front: processBatch iterates columns() more than once
      // (validation pass, then column-oriented pass), and the dictionary merges are too
      // expensive to recompute on every iteration
      List<Column> columns = new ArrayList<>();
      for (LeafReaderContext toContext : toReader.leaves()) {
        for (LeafReaderContext fromContext : fromReader.leaves()) {
          columns.addAll(mapPairOrdinals(fromContext, fromField, toContext, toField, scratch));
        }
      }
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
    }
    writer.commit();
  }

  /**
   * Merges the sorted term dictionaries of one (from-segment, to-segment) pair and returns the
   * pair's columns: the doc-map column resolving from-side doc ids to to-side doc ids, and an
   * empty companion column named with the {@code _edges} suffix. {@code scratch} is a shared
   * from-ord indexed merge buffer, safe to reuse for the next pair since the returned columns own
   * their per-doc arrays.
   */
  static List<Column> mapPairOrdinals(
      LeafReaderContext fromContext,
      String fromField,
      LeafReaderContext toContext,
      String toField,
      long[] scratch)
      throws IOException {
    SortedSetDocValues fromDV = DocValues.getSortedSet(fromContext.reader(), fromField);
    SortedSetDocValues toDV = DocValues.getSortedSet(toContext.reader(), toField);
    // map from-segment ords to to-segment ords by merging the two sorted term dictionaries
    long[] toOrdByFromOrd = scratch;
    Arrays.fill(toOrdByFromOrd, -1L);
    // dead code, kept until M:N support settles: the reverse ord map was filled but never read
    // long[] fromOrdByToOrd = new long[Math.toIntExact(toDV.getValueCount())];
    // Arrays.fill(fromOrdByToOrd, -1L);
    TermsEnum fromTerms = fromDV.termsEnum();
    TermsEnum toTerms = toDV.termsEnum();
    BytesRef fromTerm = fromTerms.next();
    BytesRef toTerm = toTerms.next();
    while (fromTerm != null && toTerm != null) {
      int cmp = fromTerm.compareTo(toTerm);
      if (cmp == 0) {
        toOrdByFromOrd[(int) fromTerms.ord()] = toTerms.ord();
        // fromOrdByToOrd[(int) toTerms.ord()] = fromTerms.ord();
        fromTerm = fromTerms.next();
        toTerm = toTerms.next();
      } else if (cmp < 0) {
        fromTerm = fromTerms.next();
      } else {
        toTerm = toTerms.next();
      }
    }
    // TODO: this degrades M:N joins to M:1. Both toDocByToOrd and toDocByFromDoc keep a single
    // to-side doc per slot, so when several to docs share a term (non-unique toField) or a from
    // doc is multi-valued with several matching terms, later assignments overwrite earlier ones
    // and only the last match survives. The read side (AIJoinQuery) already consumes all
    // docValueCount() values per doc, so only this writer needs to learn to emit multiple
    // to docs per from doc.
    int[] toDocByToOrd = new int[Math.toIntExact(toDV.getValueCount())];
    Arrays.fill(toDocByToOrd, -1);
    for (int toDoc = toDV.nextDoc();
        toDoc != DocIdSetIterator.NO_MORE_DOCS;
        toDoc = toDV.nextDoc()) {
      for (int i = 0; i < toDV.docValueCount(); i++) {
        long toOrd = toDV.nextOrd();
        toDocByToOrd[(int)toOrd] = toDoc;
      }
    }

    // resolve every from doc to its to-side ordinal: the doc's fromField ord looked up in the
    // dictionary merge result. Docs without the field, or whose term has no to-side match, keep -1
    int[] toDocByFromDoc = new int[fromContext.reader().maxDoc()];
    Arrays.fill(toDocByFromDoc, -1);
    int minFromDoc = DocIdSetIterator.NO_MORE_DOCS;
    int maxFromDoc = -1;
    int minToDoc = DocIdSetIterator.NO_MORE_DOCS;
    int maxToDoc = -1;
    for (int fromDoc = fromDV.nextDoc();
        fromDoc != DocIdSetIterator.NO_MORE_DOCS;
        fromDoc = fromDV.nextDoc()) {
      for (int i = 0; i < fromDV.docValueCount(); i++) {
        long fromOrd = fromDV.nextOrd();
        int toOrd = (int) toOrdByFromOrd[(int) fromOrd];
        if (toOrd == -1) {
          continue;
        }
        int toDoc = toDocByToOrd[toOrd];
        if (toDoc == -1) {
          continue;
        }
        toDocByFromDoc[fromDoc] = toDoc;
        minFromDoc = Math.min(minFromDoc, fromDoc);
        maxFromDoc = Math.max(maxFromDoc, fromDoc);
        minToDoc = Math.min(minToDoc, toDoc);
        maxToDoc = Math.max(maxToDoc, toDoc);
      }
    }
    String pairFieldName = pairFieldName(fromContext, fromField, toContext, toField);
    return List.of(
        ordMapBatch(
            pairFieldName,
            //fromContext, fromDV, toContext, toDV,
            toDocByFromDoc),
        edgesColumn(pairFieldName + "_fromDoc_edges", new int[]{minFromDoc,maxFromDoc}),
        edgesColumn(pairFieldName + "_toDoc_edges", new int[]{minToDoc,maxToDoc})
      );
  }

  private static LongColumn edgesColumn(String fromEdgesFieldName, int[] fromDocEdges) {
    return new LongColumn(fromEdgesFieldName, toDocsFieldType, Column.Density.SPARSE, NumericKind.INT) {
      @Override
      public LongTupleCursor tuples() {
        return new LongTupleCursor() {
          int i=-1;
          @Override
          public int nextDoc() {
            if (++i<fromDocEdges.length) {
              return 0; //try to put both vals at the doc 0
            }
            return DocIdSetIterator.NO_MORE_DOCS;
          }

          @Override
          public long longValue() {
            return (long) fromDocEdges[i];
          }
        };
      }
    };
  }

  /** The join index field name addressing the ordinal map of one (from-segment, to-segment) pair. */
  static String pairFieldName(
      LeafReaderContext fromContext, String fromField, LeafReaderContext toContext, String toField) {
    return getSideKey(fromContext, fromField) + "_" + getSideKey(toContext, toField);
  }

  /**
   * A column persisting a doc mapping: batch-local doc number is the from-side doc id and the
   * SORTED_NUMERIC docvalue is the matching to-side doc id. From docs without a match keep -1 in
   * the array and get no value, hence the column is sparse.
   */
  static Column ordMapBatch(String fieldName,
     int[] toDocByFromDoc) {

    Column column =
        new LongColumn(fieldName, toDocsFieldType, Column.Density.SPARSE, NumericKind.INT) {
          @Override
          public LongTupleCursor tuples() {
            return new LongTupleCursor() {
              private int fromDoc = -1;

              @Override
              public int nextDoc() {
                while (++fromDoc < toDocByFromDoc.length) {
                  if (toDocByFromDoc[fromDoc] >=0) {
                    return fromDoc;
                  }
                }
                return DocIdSetIterator.NO_MORE_DOCS;
              }

              @Override
              public long longValue() {
                return toDocByFromDoc[fromDoc];
              }
            };
          }
        };
    return column;
  }

  // Lucene puts no hard constraints on field names, but conservatively keep side keys usable as
  // one by reducing them to identifier characters
  private static final Pattern NON_IDENTIFIER = Pattern.compile("[^A-Za-z0-9_]");

  /**
   * Persistent identifier of one join side: the join field name, the immutable id the segment was
   * created with (it survives reopens, growing deletes mask and reorderings of {@link
   * IndexReader#leaves()}; a merge produces a new segment with a new id) and the docvalues
   * generation of the join field.
   */
  static String getSideKey(LeafReaderContext context, String field) {
    SegmentReader segmentReader = (SegmentReader) FilterLeafReader.unwrap(context.reader());
    byte[] segmentId = segmentReader.getSegmentInfo().info.getId();
    // dvGen starts at -1 and advances only when this particular field receives an in-place
    // IndexWriter.updateDocValues update; deletes only bump delGen and leave it untouched. So the
    // key is insensitive to deletes but changes when the join field's docvalues are updated.
    long dvGen = context.reader().getFieldInfos().fieldInfo(field).getDocValuesGen();
    String key = field + ":" + StringHelper.idToString(segmentId) + ":" + dvGen;
    return NON_IDENTIFIER.matcher(key).replaceAll("_");
  }

  /**
   * Allocates a directory and writes the auxiliary join index for joining {@code fromField} values
   * to {@code toField} values; see {@link writeAJoinIndex}.
   */
  public static IndexWriter writeAJoinIndex( Directory aiJoinDir,
      IndexReader fromReader, String fromField, IndexReader toReader, String toField)
      throws IOException {
    IndexWriter writer =
        new IndexWriter(
            aiJoinDir,
            new IndexWriterConfig()
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setMergeScheduler(NoMergeScheduler.INSTANCE));
    writeAJoinIndex(writer, fromReader, fromField, toReader, toField);
    return writer;
  }
}
