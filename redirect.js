(function () {
  const docsBase = "https://docs.nvidia.com/nemo/datadesigner";
  const redirectVersions = new Set(["0.5.8", "0.5.9", "0.6.0"]);
  const siteSegment = "DataDesigner";
  const devnoteSlugs = {
    "async-all-the-way-down": "async-all-the-way-down",
    "data-designer-got-skills": "data-designer-got-skills",
    "deep-research-trajectories": "deep-research-trajectories",
    "deep-research-trajectories-with-nemo-data-designer-and-mcp-tool-use":
      "deep-research-trajectories",
    "design-principles": "design-principles",
    "designing-data-designer-why-sdg-is-a-systems-problem": "design-principles",
    "engineering-an-enterprise-grade-text-to-sql-dataset-with-nemo-data-designer":
      "text-to-sql-for-nemotron-super",
    "graduate-level-science-reasoning-data-with-nemo-data-designer": "rqa-dataset",
    "have-it-your-way": "have-it-your-way",
    "have-it-your-way-customizing-data-designer-with-plugins": "have-it-your-way",
    "mitigating-prompt-sensitivity-manufacturing-robustness-through-diverse-preambles":
      "prompt-sensitivity",
    "nemotron-personas": "designing-nemotron-personas",
    "owning-the-model-stack": "owning-the-model-stack",
    "owning-the-model-stack-adaptive-concurrency-ftw": "owning-the-model-stack",
    "prompt-sensitivity": "prompt-sensitivity",
    "push-datasets-to-hugging-face-hub": "push-datasets-to-hugging-face-hub",
    "retrieval-sdg-toolkit": "retriever-sdg-toolkit",
    "retriever-sdg-toolkit": "retriever-sdg-toolkit",
    "retriever-sdg-toolkit-from-documents-to-training-data": "retriever-sdg-toolkit",
    "rqa": "rqa-dataset",
    "search-agent": "search-agent",
    "search-agent-sft-data-teaching-llms-to-browse-the-web": "search-agent",
    "structured-outputs-for-nemotron-teaching-models-to-produce-valid-json-yaml-and-xml":
      "structured-outputs-from-nemotron",
    "structured-outputs-from-nemotron": "structured-outputs-from-nemotron",
    "text-to-sql": "text-to-sql-for-nemotron-super",
    "training-a-vlm-to-understand-long-documents-an-iterative-sdg-story":
      "vlm-long-document-understanding",
    "vlm-long-document-understanding": "vlm-long-document-understanding",
  };

  function localParts() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    const siteIndex = parts.lastIndexOf(siteSegment);
    const local = siteIndex >= 0 ? parts.slice(siteIndex + 1) : parts;
    const redirectIndex = local.findIndex(
      (part) => part === "latest" || redirectVersions.has(part)
    );

    if (redirectIndex >= 0) {
      return local.slice(redirectIndex);
    }

    if (local.length === 0 || (local.length === 1 && local[0] === "index.html")) {
      return [];
    }

    return null;
  }

  function trimGeneratedFile(parts) {
    const trimmed = parts.slice();
    const last = trimmed[trimmed.length - 1];

    if (last === "index.html" || last === "404.html") {
      trimmed.pop();
    }

    return trimmed;
  }

  function joinUrl(base, parts) {
    const suffix = trimGeneratedFile(parts).join("/");

    if (!suffix) {
      return base;
    }

    return `${base.replace(/\/$/, "")}/${suffix}`;
  }

  function devnoteSlug(parts) {
    const trimmed = trimGeneratedFile(parts);

    if (trimmed.length === 0) {
      return "overview";
    }

    const [first, second, third] = trimmed;

    if (first === "archive" || first === "page") {
      return "overview";
    }

    if (first === "assets") {
      return devnoteSlugs[second] || "overview";
    }

    if (first === "posts" && second === "assets") {
      return devnoteSlugs[third] || "overview";
    }

    if (first === "posts") {
      return devnoteSlugs[second] || second || "overview";
    }

    return devnoteSlugs[first] || first;
  }

  function devnoteUrl(parts) {
    return `${docsBase}/dev-notes/${devnoteSlug(parts)}`;
  }

  function targetUrl() {
    const parts = localParts();

    if (parts === null) {
      return "";
    }

    if (parts.length === 0) {
      return docsBase;
    }

    const version = parts[0];
    const rest = parts.slice(1);

    if (rest[0] === "devnotes") {
      return devnoteUrl(rest.slice(1));
    }

    if (version === "latest") {
      return joinUrl(docsBase, rest);
    }

    if (redirectVersions.has(version)) {
      return joinUrl(`${docsBase}/v${version}`, rest);
    }

    return "";
  }

  const target = targetUrl();

  if (target) {
    window.location.replace(`${target}${window.location.search}${window.location.hash}`);
  }
})();
