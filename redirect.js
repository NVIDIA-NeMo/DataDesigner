(function () {
  const docsBase = "https://docs.nvidia.com/nemo/datadesigner";
  const redirectVersions = new Set(["0.5.8", "0.5.9", "0.6.0"]);
  const siteSegment = "DataDesigner";

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
