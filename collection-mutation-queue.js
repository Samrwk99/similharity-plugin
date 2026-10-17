/**
 * Serialize plugin-side collection mutations inside one SillyTavern process.
 * This closes races between concurrent HTTP mutations to the same physical
 * collection, including Qdrant shared tenants. It is NOT a replacement for
 * durable cross-process fencing, chat-file revisions, or exclusive jobs.
 */
export function createCollectionMutationQueue() {
    const tails = new Map();
    return {
        async acquire(key) {
            if (!key || typeof key !== 'string') throw new Error('A physical collection lock key is required');
            const previous = tails.get(key) ?? Promise.resolve();
            let signal;
            const completed = new Promise(resolve => { signal = resolve; });
            const tail = previous.catch(() => {}).then(() => completed);
            tails.set(key, tail);
            await previous.catch(() => {});
            let released = false;
            return () => {
                if (released) return;
                released = true;
                signal();
                // A later waiter may have replaced the tail for this key.
                if (tails.get(key) === tail) tails.delete(key);
            };
        },
        get pendingKeys() { return tails.size; },
    };
}

export function pluginMutationKey(req) {
    const route = String(req.path || '');
    const body = req.body || {};
    const physical = body.collectionId || body.collection || body.registryKey || body.key;
    const isChunkMutation = /^\/chunks\/(?:insert|delete|purge|ensure-eventbase-indexes|[^/]+\/(?:text|metadata))$/.test(route);
    const isProfileMutation = /^\/collection\/profile\/(?:set|manifest\/patch)$/.test(route);
    const isBackendRetarget = /^\/backend\/init\/(?:qdrant|vectra|standard)$/.test(route);
    if (!isChunkMutation && !isProfileMutation && !isBackendRetarget) return null;
    if (!physical && !isBackendRetarget) return null; // Handler validates required IDs separately.
    const user = req.user?.directories?.root || req.user?.directories?.vectors;
    if (!user) return null; // Do not pretend anonymous callers have a verified identity.
    const backend = String(isBackendRetarget ? (req.params?.backend || 'backend') : body.backend || 'standard');
    // Lock the whole *physical* Qdrant collection, not just tenant labels:
    // this is conservative but protects shared collection schema changes.
    return JSON.stringify([String(user), backend, String(physical || '__backend_configuration__')]);
}
