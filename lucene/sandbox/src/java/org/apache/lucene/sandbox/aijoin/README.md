<!--
    Licensed to the Apache Software Foundation (ASF) under one or more
    contributor license agreements.  See the NOTICE file distributed with
    this work for additional information regarding copyright ownership.
    The ASF licenses this file to You under the Apache License, Version 2.0
    (the "License"); you may not use this file except in compliance with
    the License.  You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
-->

# AI join sidecar index

`AIJoinIndex` owns an auxiliary Lucene index persisting, for every (from-segment, to-segment)
pair, a SORTED_NUMERIC column mapping from-side doc ids to to-side doc ids, plus two edges
columns with the pair's `{min, max}` doc bounds. Pair columns are named by both sides'
persistent side keys (segment id + docvalues generation of the join field), so they survive
reopens of either side and are built lazily — the first `AIJoinQuery` weight that needs a pair
writes it. See `AIJoinIndex` for the user-facing API.

## Garbage collection of dead pairs — not implemented yet

The sidecar is append-only. A side key dies when its segment is merged away, dropped, or its
join field receives an in-place docvalues update (dvGen bump). Pair columns referencing a dead
side key can never be read again, but nothing reclaims them today: the sidecar grows with every
reopen-plus-search cycle for as long as the same `AIJoinIndex` directory is reused. The
in-memory variant additionally never trims its `pairBuilds` dedup map (string keys plus
completed futures; bounded by the number of pairs ever built in the process).

Planned reaping design:

- **Batch granularity = death granularity.** Sidecar writes already go one `ColumnBatch` per
  commit so that pair-column doc numbers equal from-side doc ids (a batch must start at doc 0
  of its segment). Splitting builds further, into one batch/segment per pair — or per group of
  pairs sharing a from-segment — makes staleness map to whole sidecar segments, which
  `IndexWriter` can drop cheaply, instead of to individual fields, which it cannot.
- **Death signal.** Register `LeafReader.getCoreCacheHelper().addClosedListener(...)` on the
  from/to leaves whose side keys a build touches (the `LRUQueryCache` mechanism), marking side
  keys dead when their segment core closes. Side keys are also verifiable directly: parse them
  out of pair field names and compare against the side keys of the currently live readers.
- **Reaper.** Piggybacked on the next build (no background thread): drop every sidecar segment
  all of whose pairs reference at least one dead side key. `SearcherManager` refcounting keeps
  segment files alive for in-flight queries; `AIJoinQuery` resolves pairs by sidecar segment
  name inside single acquire/release brackets, and by construction only references pairs whose
  side segments are live, so the reaper can never pull a segment out from under a query.
- **Fallback.** When dead pairs cannot be isolated per segment (mixed batches written by older
  code), rewrite the sidecar from scratch once the dead-pair ratio crosses a threshold — the
  taxonomy-index `replaceTaxonomy` precedent.
