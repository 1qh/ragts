# Graph-Enhanced Retrieval

## Relations

Define document relationships at ingest time:

```typescript
await rag.ingest(docs, {
  embed,
  relations: {
    'doc-a': ['doc-b', 'doc-c'],
    'doc-b': ['doc-d']
  }
})
```

Relations are bidirectional — if A relates to B, querying from B also finds A.

## Typed and Weighted Relations

```typescript
await rag.ingest(docs, {
  embed,
  relations: {
    'decree-01': [
      { title: 'law-13', type: 'implements', weight: 0.9 },
      { title: 'circular-07', type: 'guided_by', weight: 0.5 }
    ],
    'law-13': ['decree-01']
  }
})
```

Plain strings default to weight 1.0.

## Graph Expansion

Enable with `graphHops` — how many relation hops to traverse:

```typescript
const { context } = await rag.query({
  query: 'construction regulations',
  embed,
  graphHops: 2
})
```

Graph-expanded chunks are scored via RRF and interleaved with vector/BM25 results.

## Tuning

| Option | Default | Effect |
|---|---|---|
| `graphHops` | disabled | Number of relation hops to traverse |
| `graphWeight` | 1 | Graph results weight in RRF (higher = more prominent) |
| `graphDecay` | 1.0 | Per-hop decay factor (0.7 = 30% less per hop) |
| `graphChunkLimit` | 200 | Max chunks from graph expansion |
| `communityBoost` | disabled | Expand from dominant community via vector similarity |

```typescript
await rag.query({
  query,
  embed,
  graphHops: 2,
  graphWeight: 1.5,
  graphDecay: 0.7,
  communityBoost: 0.5
})
```

## Community Detection

Documents connected by relations are grouped into communities via union-find. This runs automatically after ingest.

```typescript
const count = await rag.detectCommunities()
```

## Community Summaries

Generate LLM summaries per community (stored as searchable documents):

```typescript
await rag.buildCommunitySummaries({
  embed,
  minCommunitySize: 2,
  summarize: async (docs) => {
    const { text } = await generateText({
      model: chat,
      prompt: docs.map(d => d.content).join('\n')
    })
    return text
  }
})
```

## Global Query

Map-reduce across community summaries:

```typescript
const { answer, partialAnswers } = await rag.globalQuery({
  embed,
  query: 'What are the key themes?',
  generate: async (context, query) => {
    const { text } = await generateText({ model: chat, prompt: context, system: query })
    return text
  },
  maxCommunities: 10
})
```

Each community produces a partial answer, then all partial answers are merged into a final answer.
