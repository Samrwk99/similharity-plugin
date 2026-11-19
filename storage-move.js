/**
 * Step 9d / P6: server-side copy of ONE collection between two layouts of the
 * SAME backend, and removal of an explicitly named copy.
 *
 * The extension orchestrates the move (lease, saved settings, strict commit of
 * the locator, identity ledger); this module only moves vectors:
 *   copy:        read every vector of the source location (no re-embedding),
 *                write them to the target layout, verify (same count, 5 random
 *                hashes with identical vectors). Any failure removes what was
 *                written to the target, scoped to this collection.
 *   delete-copy: remove this collection's data (and its saved-settings slot)
 *                from one named location. Shared stores are never dropped.
 *
 * Legacy Qdrant multitenancy is refused as a source (its contents are
 * incomplete: identical text in two chats shared one point).
 */

import { createHash } from 'node:crypto';
import {
    MT_V2_NAME, MILVUS_OWN_NAME, milvusOwnName, mtV2Name, configKeyForTenant,
} from './tenancy.js';

const QDRANT_LEGACY_MT = 'vecthare_multitenancy';
const MILVUS_LEGACY_SHARED = 'vecthare_main';
const MAX_MILVUS_ROWS = 16000; // Milvus query window (offset + limit <= 16384)
const legacyConfigKey = id => `vecthare_initial_${createHash('sha256').update(id).digest('hex')}`;

const fail = (message, status = 400) => Object.assign(new Error(message), { status });

/**
 * Validates a location named by the extension for this collection.
 * @returns {{layout:string, physical:string|null, dim:number|null}}
 */
export function parseLocation(backend, collectionId, tenantKey, location, { allowPendingDim = false } = {}) {
    if (!location || typeof location !== 'object') throw fail('location required');
    const { layout } = location;
    const dim = location.dim == null ? null : Number(location.dim);
    if (dim !== null && (!Number.isSafeInteger(dim) || dim < 1 || dim > 65536)) throw fail('Invalid dim');
    if (backend === 'qdrant') {
        if (layout === 'per-collection') return { layout, physical: collectionId, dim };
        if (layout === 'legacy-mt') return { layout, physical: QDRANT_LEGACY_MT, dim };
        if (layout === 'mt-v2') {
            const physical = location.physical ?? (dim ? mtV2Name(dim) : null);
            if (physical === null && allowPendingDim) return { layout, physical: null, dim: null };
            if (!MT_V2_NAME.test(physical || '')) throw fail('Invalid v2 location');
            const fromName = Number(MT_V2_NAME.exec(physical)[1]);
            if (dim !== null && dim !== fromName) throw fail('Location name does not match its dim');
            return { layout, physical, dim: fromName };
        }
    } else if (backend === 'milvus') {
        if (layout === 'legacy-shared') return { layout, physical: MILVUS_LEGACY_SHARED, dim };
        if (layout === 'per-collection') {
            const physical = milvusOwnName(tenantKey);
            if (location.physical && location.physical !== physical) throw fail('Location does not belong to this collection');
            return { layout, physical, dim };
        }
        if (layout === 'mt-v2') {
            const physical = location.physical ?? (dim ? mtV2Name(dim) : null);
            if (physical === null && allowPendingDim) return { layout, physical: null, dim: null };
            if (!MT_V2_NAME.test(physical || '')) throw fail('Invalid v2 location');
            return { layout, physical, dim: Number(MT_V2_NAME.exec(physical)[1]) };
        }
    }
    throw fail(`Unsupported ${backend} layout ${layout}`);
}

const close = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length &&
    a.every((v, i) => Math.abs(Number(v) - Number(b[i])) <= 1e-6 * Math.max(1, Math.abs(Number(v))));

/** Operations on one location of one collection (vectors only). */
function locationOps(ctx, backend, loc, tenant, collectionId) {
    const { qdrant, milvus } = ctx;
    if (backend === 'qdrant') {
        const scope = loc.layout === 'mt-v2' ? { tenantKey: tenant.tenantKey } : {};
        return {
            exists: async () => loc.physical ? await qdrant.collectionExists(loc.physical) : false,
            count: async () => {
                if (!loc.physical || !await qdrant.collectionExists(loc.physical)) return 0;
                return loc.layout === 'mt-v2'
                    ? await qdrant.countTenant(loc.physical, tenant.tenantKey)
                    : (await qdrant._request('POST', `/collections/${encodeURIComponent(loc.physical)}/points/count`, { exact: true })).result.count;
            },
            readAll: async () => await qdrant.listItems(loc.physical, scope, { includeVectors: true, strictRead: true }),
            vectorOf: async hash => (await qdrant.getItem(loc.physical, hash, scope))?.vector ?? null,
            configKey: loc.layout === 'mt-v2' ? configKeyForTenant(tenant.tenantKey) : legacyConfigKey(collectionId),
            readConfig: async key => loc.physical && await qdrant.collectionExists(loc.physical)
                ? await qdrant.readInitialCollectionConfig(loc.physical, key) : undefined,
            write: async (items, embedding) => {
                for (let i = 0; i < items.length; i += 100) {
                    const batch = items.slice(i, i + 100);
                    if (loc.layout === 'mt-v2') {
                        await qdrant.insertV2(loc.physical, batch, tenant, embedding);
                    } else {
                        await qdrant.insertVectors(loc.physical, batch.map(item => {
                            const metadata = { ...(item.metadata || {}) };
                            delete metadata.tenantKey; delete metadata.user; delete metadata.content_type;
                            return { ...item, metadata };
                        }), { embeddingSource: embedding.embeddingSource, embeddingModel: embedding.embeddingModel });
                    }
                }
            },
            // Removes this collection's data at this location. A per-collection
            // store is its own and is deleted with its settings; in a shared
            // store only this tenant's points go and its settings slot is emptied.
            remove: async () => {
                if (!loc.physical || !await qdrant.collectionExists(loc.physical)) return;
                if (loc.layout === 'per-collection') await qdrant.purgeAll(loc.physical);
                else if (loc.layout === 'mt-v2') await qdrant.purgeV2(loc.physical, tenant.tenantKey, configKeyForTenant(tenant.tenantKey));
                else throw fail('The old shared Qdrant store cannot be cleaned per collection', 409);
            },
        };
    }
    const legacy = loc.layout === 'legacy-shared';
    const scopeTenant = { type: tenant.type, sourceId: tenant.sourceId };
    return {
        exists: async () => loc.physical ? (await milvus.listCollectionNames()).includes(loc.physical) : false,
        count: async () => {
            if (!loc.physical || !(await milvus.listCollectionNames()).includes(loc.physical)) return 0;
            return legacy ? await milvus.countItems(scopeTenant) : await milvus.countV2(loc.physical, tenant.tenantKey);
        },
        readAll: async () => legacy
            ? await milvus.listItems(MILVUS_LEGACY_SHARED, scopeTenant, { includeVectors: true, limit: 16384 })
            : await milvus.listV2(loc.physical, tenant.tenantKey, { includeVectors: true, limit: 16384 }),
        vectorOf: async hash => legacy
            ? (await milvus.getItem(MILVUS_LEGACY_SHARED, hash, scopeTenant))?.vector ?? null
            : (await milvus.getByHashesV2(loc.physical, tenant.tenantKey, [hash], { withVector: true }))[0]?.vector ?? null,
        configKey: legacy ? legacyConfigKey(collectionId) : configKeyForTenant(tenant.tenantKey),
        readConfig: async key => {
            if (!loc.physical || !(await milvus.listCollectionNames()).includes(loc.physical)) return undefined;
            return legacy ? await milvus.readInitialCollectionConfig(key) : await milvus.readConfigV2(loc.physical, key);
        },
        write: async (items, embedding) => {
            for (let i = 0; i < items.length; i += 500) {
                await milvus.insertV2(loc.physical, items.slice(i, i + 500), tenant, embedding, { partitionKey: loc.layout === 'mt-v2' });
            }
        },
        remove: async () => {
            if (!loc.physical || !(await milvus.listCollectionNames()).includes(loc.physical)) return;
            if (legacy) {
                const purged = await milvus.purgeCollection(MILVUS_LEGACY_SHARED, scopeTenant);
                await milvus.tombstoneLegacyConfig(legacyConfigKey(collectionId));
                return purged; // VH-011: { leftShared }
            } else {
                await milvus.purgeV2(loc.physical, tenant.tenantKey, { drop: loc.layout === 'per-collection', configKey: configKeyForTenant(tenant.tenantKey) });
            }
        },
    };
}

/**
 * Copies one collection from `fromRaw` to the layout `toRaw.layout`.
 * @returns {Promise<{copied:number, sourceCount:number, target:{layout,physical,dim}}>}
 */
export async function copyCollection(ctx, { backend, collectionId, tenant, fromRaw, toRaw, embedding = {} }) {
    const from = parseLocation(backend, collectionId, tenant.tenantKey, fromRaw);
    let to = parseLocation(backend, collectionId, tenant.tenantKey, toRaw, { allowPendingDim: true });
    if (backend === 'qdrant' && from.layout === 'legacy-mt') {
        throw fail('The old shared Qdrant store cannot be moved (its contents are incomplete); re-vectorize instead', 409);
    }
    if (from.layout === to.layout) throw fail('Source and target layout are the same');
    if (backend === 'milvus' && to.layout === 'legacy-shared') throw fail('Moving into the old shared Milvus store is not offered');
    if (backend === 'qdrant' && to.layout === 'legacy-mt') throw fail('Moving into the old shared Qdrant store is not offered');

    const source = locationOps(ctx, backend, from, tenant, collectionId);
    if (!await source.exists()) throw fail('The source location holds nothing for this collection', 409);
    const sourceCount = await source.count();
    if (backend === 'milvus' && sourceCount > MAX_MILVUS_ROWS) {
        throw fail(`This collection has ${sourceCount} vectors; the move tool handles up to ${MAX_MILVUS_ROWS} on Milvus. Re-vectorize instead.`, 409);
    }
    const items = await source.readAll();
    if (!Array.isArray(items) || items.length !== sourceCount) {
        throw fail(`The source could not be read completely (${items?.length ?? 0}/${sourceCount}); nothing was copied`, 409);
    }
    if (!items.length) throw fail('The source collection is empty; nothing to move', 409);
    if (items.some(item => !Array.isArray(item.vector) || !item.vector.length)) {
        throw fail('Some source vectors could not be read; nothing was copied', 409);
    }
    const dim = items[0].vector.length;
    if (to.layout === 'mt-v2') to = { ...to, physical: mtV2Name(dim), dim };
    else to = { ...to, dim };

    const target = locationOps(ctx, backend, to, tenant, collectionId);
    // Never write over another copy: the target must hold nothing for this
    // collection, neither vectors nor live saved settings.
    if (await target.count() > 0) {
        throw fail('The target already holds a copy of this collection (an old copy?). Delete it first; nothing was copied.', 409);
    }
    if (await target.readConfig(target.configKey) !== undefined) {
        throw fail('The target already holds saved settings for this collection. Delete the old copy first; nothing was copied.', 409);
    }

    let wrote = false;
    try {
        wrote = true;
        await target.write(items, { embeddingSource: embedding.source, embeddingModel: embedding.model });
        const unique = new Set(items.map(item => String(item.hash))).size;
        const copied = await target.count();
        if (copied !== unique) throw fail(`Verification failed: ${copied} of ${unique} vectors in the target`, 500);
        // Five random hashes must come back with identical vectors.
        const pool = [...items];
        for (let i = 0; i < Math.min(5, pool.length); i++) {
            const j = i + Math.floor(Math.random() * (pool.length - i));
            [pool[i], pool[j]] = [pool[j], pool[i]];
            const back = await target.vectorOf(pool[i].hash);
            if (!close(back, pool[i].vector)) throw fail(`Verification failed: vector of hash ${pool[i].hash} differs`, 500);
        }
        return { copied, sourceCount, target: { layout: to.layout, physical: to.physical, dim: to.dim } };
    } catch (error) {
        if (wrote) {
            try { await target.remove(); }
            catch (cleanup) { error.message += ` (and the partial copy could not be removed: ${cleanup.message})`; }
        }
        throw error;
    }
}

/** Removes this collection's data and saved settings from ONE named location. */
export async function deleteCopy(ctx, { backend, collectionId, tenant, locationRaw }) {
    const loc = parseLocation(backend, collectionId, tenant.tenantKey, locationRaw);
    const ops = locationOps(ctx, backend, loc, tenant, collectionId);
    const removed = await ops.remove();
    // VH-011: in Milvus' older shared store, rows whose text another collection
    // also stores have to stay; everything else must be gone.
    if (backend === 'milvus' && loc.layout === 'legacy-shared') {
        const { unshared, shared } = await ctx.milvus.countUnsharedItems({ type: tenant.type, sourceId: tenant.sourceId });
        if (unshared !== 0) throw fail(`The copy could not be removed completely (${unshared} vectors remain)`, 500);
        return { removed: true, ...(shared ? { leftShared: shared } : {}) };
    }
    const left = await ops.count();
    if (left !== 0) throw fail(`The copy could not be removed completely (${left} vectors remain)`, 500);
    return { removed: true, ...(removed?.leftShared ? { leftShared: removed.leftShared } : {}) };
}
