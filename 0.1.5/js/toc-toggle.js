// Wrap in a check to ensure document$ exists
if (typeof document$ !== "undefined") {
    document$.subscribe(function() {
        // Check if this is a Code Reference page (contains mkdocstrings content)
        const isCodeReferencePage = document.querySelector(".doc.doc-contents");

        // Check if this is a Concepts page (URL contains /concepts/)
        const isConceptsPage = window.location.pathname.includes("/concepts/");

        // Check if this is a Plugins page (URL contains /plugins/)
        const isPluginsPage = window.location.pathname.includes("/plugins/");

        if (isCodeReferencePage || isConceptsPage || isPluginsPage) {
            // Show TOC for Code Reference, Concepts, and Plugins pages by adding class to body
            document.body.classList.add("show-toc");
            console.log("Code Reference, Concepts, or Plugins page detected - showing TOC");
        } else {
            // Hide TOC for all other pages by removing class from body
            document.body.classList.remove("show-toc");
            console.log("Non-Code Reference/Concepts/Plugins page - hiding TOC");
        }
    });
} else {
    console.error("document$ observable not found - Material theme may not be loaded");
}
