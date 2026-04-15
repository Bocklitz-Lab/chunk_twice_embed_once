export {}; // treat file as module

declare const self: ServiceWorkerGlobalScope;

const SHELL_CACHE = 'world-clock-shell-v1';
const OFFLINE_URLS = ['/'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(OFFLINE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== SHELL_CACHE).map((key) => caches.delete(key)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      const networkFetch = fetch(event.request)
        .then((response) => {
          const copy = response.clone();
          caches.open(SHELL_CACHE).then((cache) => cache.put(event.request, copy));
          return response;
        })
        .catch(() => cached ?? caches.match('/'));

      return cached ?? networkFetch;
    })
  );
});

self.addEventListener('message', (event) => {
  if (event.data?.type === 'invalidate-cache') {
    caches.delete(SHELL_CACHE).then(() => caches.open(SHELL_CACHE));
  }
});
