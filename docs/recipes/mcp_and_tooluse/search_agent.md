# Nemotron Super Search Agent

!!! note "Dev Note"
    For a deep dive into the pipeline design, production yield analysis, correctness challenges, and key takeaways, see [Search Agent SFT Data: Teaching LLMs to Browse the Web](../../../devnotes/search-agent-sft-data-teaching-llms-to-browse-the-web/).

!!! tip "Seed Dataset"
    This recipe includes built-in demo seeds (3 Wikidata knowledge graph paths) for quick testing. For production use, generate your own seed dataset from Wikidata random walks -- the dev note above describes the seed generation process (SPARQL queries, anti-meta filters, hop range 4-8). Each seed row needs: `seed_entity`, `final_answer_entity`, `readable_path`, `num_hops_in_graph`, and `ground_truth`. Pass your seed file via `--seed-path`.

[Download Code :octicons-download-24:](../../../assets/recipes/mcp_and_tooluse/search_agent.py){ .md-button download="search_agent.py" }

```python
--8<-- "assets/recipes/mcp_and_tooluse/search_agent.py"
```
