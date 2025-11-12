// Wrap in a check to ensure document$ exists
if (typeof document$ !== "undefined") {
    document$.subscribe(function() {
        // Check if this is a Code Reference page (contains mkdocstrings content)
        const isCodeReferencePage = document.querySelector(".doc.doc-contents");

        if (isCodeReferencePage) {
            // Show TOC for Code Reference pages by adding class to body
            document.body.classList.add("show-toc");
            console.log("Code Reference page detected - showing TOC");
        } else {
            // Hide TOC for all other pages by removing class from body
            document.body.classList.remove("show-toc");
            console.log("Non-Code Reference page - hiding TOC");
        }
    });
} else {
    console.error("document$ observable not found - Material theme may not be loaded");
}
