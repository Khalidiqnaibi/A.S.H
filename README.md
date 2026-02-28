# A.S.H

## Archi : 
``` php
ASH
 ├── Classification Engine (deterministic)
 ├── Tool Execution Layer
 ├── Memory System
 │     ├── CoreMemory (immutable)
 │     ├── EpisodicMemory (time based)
 │     ├── EntityMemory (structured people/things)
 │     └── MemoryRouter (logic)
 ├── Emotion Engine
 └── LLM Narrator
```

### New memo archi:
``` php
run()
 ├─ MemoryRouter.preprocess(query)
 │    ├─ NER extraction
 │    ├─ Decide entity vs episodic vs core routing
 │
 ├─ CoreMemory.retrieve()        # constraints, standards
 ├─ EpisodicMemory.retrieve()    # recent relevant events
 ├─ EntityMemory.retrieve()      # specific person/thing memory
 │
 ├─ classify_and_route()
 ├─ deterministic tool execution
 │
 ├─ MemoryRouter.postprocess()
 │    ├─ Store episodic memory
 │    ├─ Update entity memory
 │
 ├─ render_with_llm(context + facts + memory)
 └─ append history
```
### prev memo archi:

``` bash
chroma run --host localhost --port 2000 --path ./chroma_data
```

