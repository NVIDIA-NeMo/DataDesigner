// Wrap in a check to ensure document$ exists
if (typeof document$ !== "undefined") {
    document$.subscribe(function() {
        // Check if this is a Concepts page (URL contains /concepts/)
        const isConceptsPage = window.location.pathname.includes("/concepts/");

        // Check if this is a Plugins page (URL contains /plugins/)
        const isPluginsPage = window.location.pathname.includes("/plugins/");

        if (isConceptsPage || isPluginsPage) {
            // Show TOC for Concepts and Plugins pages by adding class to body
            document.body.classList.add("show-toc");
            console.log("Concepts or Plugins page detected - showing TOC");
        } else {
            // Hide TOC for all other pages by removing class from body
            document.body.classList.remove("show-toc");
            console.log("Non-Concepts/Plugins page - hiding TOC");
        }
    });
} else {
    console.error("document$ observable not found - Material theme may not be loaded");
}
