window.addEventListener('load', () => {
    const app = new URLSearchParams(window.location.search).get('app');
    if (app === 'asset') void switchTab('asset');
});
