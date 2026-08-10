document.querySelectorAll(".progress-bar[data-value]").forEach((bar) => {
    const value = Number(bar.dataset.value);
    bar.style.width = `${Math.max(0, Math.min(value || 0, 100))}%`;
});
