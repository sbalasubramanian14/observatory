"""Hand-labelled clustering corpus. (story_label, outlet, headline + lead).

Labels A-D are multi-outlet stories; S* are singletons. Grown over time from
real misclusterings observed in production - every fix should add its case.
"""

CORPUS: list[tuple[str, str, str]] = [
    ("A", "TechCrunch", "DeepSeek releases V4, an open-weights MoE model it claims matches frontier systems. The Chinese lab published weights under an MIT license on Hugging Face early Tuesday."),
    ("A", "TheVerge", "DeepSeek's new V4 model is free to download and claims GPT-class performance. The release lands under a permissive license, unusual for a model of this scale."),
    ("A", "VentureBeat", "Chinese AI lab DeepSeek open-sources V4 mixture-of-experts model with 1.2T parameters. Benchmarks published alongside the release claim parity with closed frontier models on coding tasks."),
    ("A", "Reuters", "DeepSeek publishes new AI model weights publicly, escalating open-source competition. The move intensifies pressure on US labs that keep their frontier weights private."),
    ("A", "HackerNews", "Show HN: DeepSeek V4 weights are up on HuggingFace. Running it locally on 2x4090 with 4-bit quantization, quality seems genuinely close to Opus for refactoring work."),
    ("A", "ArsTechnica", "DeepSeek V4 arrives with open weights and bold benchmark claims. Independent evaluation has not yet confirmed the lab's reported SWE-bench numbers."),
    ("B", "Politico", "EU delays enforcement of AI Act high-risk provisions by eighteen months. The Commission cited unfinished technical standards as the reason for the postponement."),
    ("B", "Reuters", "European Commission postpones key AI Act obligations until 2028. Industry groups had lobbied heavily for additional implementation time."),
    ("B", "EURACTIV", "Brussels pushes back AI Act Article 6 deadline amid standards delay. Civil society organisations criticised the decision as capitulation to industry pressure."),
    ("B", "FT", "Brussels grants AI companies extra time to comply with landmark rules. The delay affects obligations for systems classified as high-risk under the regulation."),
    ("C", "CNBC", "Nvidia beats Q3 estimates as datacenter revenue climbs 62% year over year. The company guided above consensus for the coming quarter."),
    ("C", "Bloomberg", "Nvidia's datacenter sales surge again, topping analyst expectations. Shares rose in after-hours trading following the report."),
    ("C", "WSJ", "Nvidia quarterly results exceed forecasts on continued AI infrastructure demand. Executives said supply constraints are easing but remain a factor."),
    ("D", "arXiv", "Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents. We introduce a learned consolidator that rewrites regions of agent memory during idle compute."),
    ("D", "TwitterThread", "New paper: Auto-Dreamer does offline memory consolidation for LLM agents. Treats a memory region as read-only evidence then synthesizes a compact replacement. Nice results on long-horizon tasks."),
    ("D", "MarkTechPost", "Researchers propose Auto-Dreamer for agent memory consolidation during idle time. The method abstracts across sessions to replace bloated memory regions with compact summaries."),
    ("S1", "Anthropic", "Introducing improvements to Claude Code hooks and plugin configuration. Teams can now define lifecycle hooks that run before and after tool calls."),
    ("S2", "IEEE", "Autonomous chemistry lab at Argonne runs 24-hour discovery loops for battery materials. The facility combines robotic synthesis with model-driven hypothesis generation."),
    ("S3", "TheInformation", "OpenAI in talks to raise at a higher valuation, sources say. The round would value the company well above its previous mark."),
    ("S4", "arXiv", "Sparse autoencoders fail to recover ground-truth features in synthetic settings. We construct toy models where the true features are known and evaluate recovery rates."),
    ("S5", "DataCenterDynamics", "Texas grid operator warns of capacity shortfall from datacenter buildout. Interconnection queues have grown substantially over the past eighteen months."),
    ("S6", "GitHub", "Release v2.0 of a popular vector database adds hybrid search and metadata filtering. The update focuses on recall improvements for mixed keyword and semantic queries."),
]
