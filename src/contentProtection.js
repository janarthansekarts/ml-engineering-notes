// Content protection - disable right-click and text selection for published content
if (typeof window !== 'undefined') {
  document.addEventListener('contextmenu', (e) => e.preventDefault());
  document.addEventListener('selectstart', (e) => {
    if (e.target.closest('pre, code')) return;
    e.preventDefault();
  });
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && (e.key === 'u' || e.key === 's' || e.key === 'p')) {
      e.preventDefault();
    }
  });
}
