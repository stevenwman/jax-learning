window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset math after Material's instant navigation swaps page content
document$.subscribe(() => {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise();
  }
});
