/**
 * Step 9a / P1: read-only storage report.
 *
 * For every collection VectHare's own metadata file (collections.json) knows
 * about, this works out where its vectors physically are, how many there are,
 * and where its saved original settings ("initial config") were found. It also
 * lists the shared physical stores and the VectHare-named physical stores that
 * no record points at.
 *
 * READ-ONLY. It never creates, changes or deletes a collection, a point, a
 * file or any VectHare metadata. Backends that are not connected on the server
 * are reported as such and never connected from here.
 *
 * Everything is injected (backends, file system, helpers), so this module has
 * no heavy imports and can be exercised by probes without Qdrant/Milvus/LanceDB.
 */

import path from 'node:path';
import fsDefault from 'node:fs/promises';
import { createHash } from 'node:crypto';

export const STORAGE_REPORT_VERSION = 1;

const QDRANT_LEGACY_MT = 'vecthare_multitenancy';
const MILVUS_LEGACY_SHARED = 'vecthare_main';
const SHARED_NAME = /^(vecthare_main|vecthare_multitenancy|vecthare_mt_\d+)$/;
const REMOTE_BACKENDS = ['qdrant', 'milvus'];
const KNOWN_BACKENDS = ['standard', 'vectra', 'lancedb', 'qdrant', 'milvus'];
const MAX_IDS = 5000;
const MAX_MT_SCROLL = 500000;
const MAX_VECTRA_INDEX_BYTES = 512 * 1024 * 1024;

export const initialConfigKey = id =>
    `vecthare_initial_${createHash('sha256').update(id).digest('hex')}`;
/** 9c: v2 layouts key saved settings by tenant, not by logical id. */
const initialConfigKeyFor = tenantKey => initialConfigKey(tenantKey);
const MT_V2 = /^vecthare_mt_[1-9]\d{0,5}$/;

const isObject = v => v !== null && typeof v === 'object' && !Array.isArray(v);
const normBackend = b => (b === 'vectra' ? 'standard' : b);
const safeToken = s => typeof s === 'string' && /^[A-Za-z0-9_.-]+$/.test(s) && s !== '.' && s !== '..';

/**
 * Splits a stored key into {backend, source, collectionId}, the same three
 * shapes the extension writes: `backend:source:id`, `source:id`, `id`.
 */
export function parseStoredKey(key) {
    if (typeof key !== 'string' || !key) return null;
    let m = /^(standard|vectra|lancedb|qdrant|milvus):([A-Za-z0-9_-]+):(.+)$/s.exec(key);
    if (m) return { backend: normBackend(m[1]), source: m[2], collectionId: m[3] };
    m = /^([A-Za-z0-9_-]+):(vecthare_.+)$/s.exec(key);
    if (m) return { backend: null, source: m[1], collectionId: m[2] };
    return { backend: null, source: null, collectionId: key };
}

/**
 * Logical collections named by collections.json (registry, collection records,
 * chat bindings), each with the backends/sources it is recorded against.
 * @returns {Map<string, {hints:Set<string>, backends:Map<string, Set<string>>}>}
 */
export function collectRecordedCollections(state) {
    const out = new Map();
    const add = (raw, hint, backend = null, source = null, locator = undefined) => {
        const parsed = parseStoredKey(raw);
        if (!parsed || !parsed.collectionId || parsed.collectionId.length > 255) return;
        const id = parsed.collectionId;
        const entry = out.get(id) || { hints: new Set(), backends: new Map(), locator: null, oldCopies: [] };
        if (locator !== undefined && isObject(locator)) entry.locator = locator;
        entry.hints.add(hint);
        const b = normBackend(backend || parsed.backend);
        const s = source || parsed.source;
        if (b && KNOWN_BACKENDS.includes(b)) {
            const sources = entry.backends.get(b) || new Set();
            if (s) sources.add(s);
            entry.backends.set(b, sources);
        }
        out.set(id, entry);
    };
    if (!isObject(state)) return out;
    for (const key of Array.isArray(state.registry) ? state.registry : []) add(key, 'registry');
    for (const [key, record] of Object.entries(isObject(state.collections) ? state.collections : {})) {
        add(key, 'metadata', null, null, isObject(record) ? record.storage : undefined);
        // 9d: copies left behind by a storage move ("old copy (moved)").
        const parsed = parseStoredKey(key);
        if (parsed && Array.isArray(record?.storageOldCopies)) {
            const entry = out.get(parsed.collectionId);
            for (const copy of record.storageOldCopies) {
                // VH-010: a pending copy (the target of an unfinished move) may not
                // know its physical store yet; it is matched by layout below.
                const pendingUnknown = isObject(copy) && copy.pending === true && copy.physical == null;
                if (isObject(copy) && (typeof copy.physical === 'string' || pendingUnknown) && typeof copy.layout === 'string') {
                    entry.oldCopies.push({ backend: normBackend(copy.backend), layout: copy.layout, physical: copy.physical ?? null,
                        ...(copy.pending === true ? { pending: true } : {}),
                        ...(Number.isSafeInteger(copy.dim) ? { dim: copy.dim } : {}), ...(copy.movedAt ? { movedAt: copy.movedAt } : {}),
                        ...(copy.startedAt ? { startedAt: copy.startedAt } : {}) });
                }
            }
        }
    }
    const bindings = state.chatCollectionBindings?.byCollection;
    if (isObject(bindings)) {
        for (const [key, rec] of Object.entries(bindings)) {
            add(key, 'binding');
            for (const store of Array.isArray(rec?.stores) ? rec.stores : []) {
                if (typeof store?.backend === 'string') {
                    add(key, 'binding', store.backend, typeof store.source === 'string' ? store.source : null);
                }
            }
        }
    }
    return out;
}

/**
 * Counts SillyTavern user folders next to this user's folder. Unknown -> the
 * caller treats the server as multi-user and says it is not confirmed.
 */
export async function countServerUsers(dataRoot, fs = fsDefault) {
    try {
        const entries = await fs.readdir(dataRoot, { withFileTypes: true });
        let users = 0;
        for (const entry of entries) {
            if (!entry.isDirectory() || entry.name.startsWith('_') || entry.name.startsWith('.')) continue;
            const has = await fs.access(path.join(dataRoot, entry.name, 'settings.json')).then(() => true, () => false);
            if (has) users++;
        }
        if (users < 1) return { mode: 'multi-user', users: null, confirmed: false };
        return { mode: users > 1 ? 'multi-user' : 'single-user', users, confirmed: true };
    } catch {
        return { mode: 'multi-user', users: null, confirmed: false };
    }
}

function checkRecord(record, id, expectMultitenancy, expectLayout) {
    if (record === undefined) return { status: 'missing' };
    let value = record;
    if (typeof value === 'string') {
        try { value = JSON.parse(value); } catch { return { status: 'invalid', reason: 'cannot be decoded' }; }
    }
    if (isObject(value) && value.vecthareTombstone === 1) return { status: 'missing' };
    if (!isObject(value) || value.collectionId !== id || value.snapshot?.schemaVersion !== 1) {
        return { status: 'invalid', reason: 'malformed or belongs to another collection' };
    }
    const storage = value.snapshot?.storage || {};
    if (expectMultitenancy !== undefined && (storage.multitenancy === true) !== expectMultitenancy) {
        return { status: 'invalid', reason: `records multitenancy ${storage.multitenancy === true ? 'on' : 'off'} for this location` };
    }
    if ((storage.layout ?? null) !== (expectLayout ?? null)) {
        return { status: 'invalid', reason: `records layout ${storage.layout ?? 'legacy'} for this location` };
    }
    return {
        status: 'found',
        ...(typeof value.owner === 'string' ? { owner: value.owner } : {}),
        ...(isObject(value.snapshot.movedFrom) ? { movedFrom: value.snapshot.movedFrom } : {}),
        recordedAt: typeof value.snapshot.recordedAt === 'string' ? value.snapshot.recordedAt : null,
        provider: value.snapshot.embedding?.provider ?? null,
        model: value.snapshot.embedding?.model ?? null,
        pipeline: value.snapshot.chunking?.pipeline ?? null,
    };
}

/** A collection's Step 9b locator, as recorded (validity checked loosely here). */
function summarizeLocator(locator) {
    if (!isObject(locator)) return null;
    const backend = normBackend(locator.backend);
    const valid = locator.locatorVersion === 1 && typeof locator.physical === 'string' &&
        typeof locator.layout === 'string' && KNOWN_BACKENDS.includes(backend);
    return { backend, layout: String(locator.layout ?? ''), physical: String(locator.physical ?? ''), valid };
}

/** found/missing/multiple/invalid from per-location results. */
function summarizeConfigs(locations) {
    const found = locations.filter(l => l.config?.status === 'found');
    const invalid = locations.filter(l => l.config?.status === 'invalid');
    const unavailable = locations.filter(l => l.config?.status === 'unavailable');
    if (found.length > 1) return { status: 'multiple', locations: found.map(l => l.physical) };
    if (found.length === 1) {
        return { status: 'found', location: found[0].physical, ...found[0].config,
            ...(invalid.length ? { alsoInvalidAt: invalid.map(l => l.physical) } : {}) };
    }
    if (invalid.length) return { status: 'invalid', locations: invalid.map(l => l.physical), reason: invalid[0].config.reason };
    if (unavailable.length) return { status: 'unavailable', reason: unavailable[0].config.reason };
    return { status: 'missing' };
}

// ---------------------------------------------------------------------------
// Qdrant
// ---------------------------------------------------------------------------

async function qdrantInfo(qdrant, name) {
    const info = await qdrant._request('GET', `/collections/${encodeURIComponent(name)}`);
    const config = info?.result?.config;
    if (!isObject(config)) throw new Error(`Qdrant returned no description for ${name}`);
    const vectors = config.params?.vectors;
    const dim = Number.isSafeInteger(vectors?.size) ? vectors.size : null;
    const metadata = isObject(config.metadata) ? config.metadata : {};
    return { dim, metadata, points: Number.isSafeInteger(info.result.points_count) ? info.result.points_count : null };
}

async function qdrantCount(qdrant, name, filter = null) {
    const res = await qdrant._request('POST', `/collections/${encodeURIComponent(name)}/points/count`,
        { exact: true, ...(filter ? { filter } : {}) });
    const count = res?.result?.count;
    if (!Number.isSafeInteger(count) || count < 0) throw new Error(`Invalid Qdrant count for ${name}`);
    return count;
}

/** Legacy multitenancy: points per content_type (and without one). */
async function qdrantTallyContentTypes(qdrant, name) {
    const byType = new Map();
    let withoutType = 0;
    let scanned = 0;
    let offset = null;
    let truncated = false;
    const seen = new Set();
    do {
        const body = { limit: 1000, with_payload: { include: ['content_type'] }, with_vector: false };
        if (offset !== null) body.offset = offset;
        const res = await qdrant._request('POST', `/collections/${encodeURIComponent(name)}/points/scroll`, body);
        const points = res?.result?.points;
        if (!Array.isArray(points)) throw new Error(`Invalid Qdrant scroll response for ${name}`);
        for (const point of points) {
            const type = point?.payload?.content_type;
            if (typeof type === 'string' && type) byType.set(type, (byType.get(type) || 0) + 1);
            else withoutType++;
        }
        scanned += points.length;
        offset = res.result.next_page_offset ?? null;
        if (offset !== null) {
            const key = JSON.stringify(offset);
            if (seen.has(key)) { truncated = true; break; }
            seen.add(key);
        }
        if (scanned >= MAX_MT_SCROLL && offset !== null) { truncated = true; break; }
    } while (offset !== null);
    return { byType, withoutType, scanned, truncated };
}

async function qdrantSection(ctx, ids, report) {
    const qdrant = ctx.qdrant;
    const names = new Set(await qdrant.listCollectionNames());
    const legacy = { exists: names.has(QDRANT_LEGACY_MT) };
    if (legacy.exists) {
        try {
            legacy.info = await qdrantInfo(qdrant, QDRANT_LEGACY_MT);
            legacy.tally = await qdrantTallyContentTypes(qdrant, QDRANT_LEGACY_MT);
        } catch (error) {
            legacy.error = error.message;
        }
    }

    // 9c: v2 stores (vecthare_mt_<dim>): read each one's saved-settings map once.
    const v2Stores = [];
    for (const name of [...names].filter(n => MT_V2.test(n)).sort()) {
        try { v2Stores.push({ name, ...await qdrantInfo(qdrant, name) }); }
        catch (error) { v2Stores.push({ name, error: error.message }); }
    }

    const rows = [];
    for (const id of ids) {
        const row = { collectionId: id, backend: 'qdrant' };
        const locations = [];
        try {
            const tenantKey = ctx.tenantKeyOf?.(id);
            const pointer = ctx.locators?.get(id);
            if (tenantKey) {
                for (const store of v2Stores) {
                    if (store.error) continue;
                    const config = checkRecord(store.metadata[initialConfigKeyFor(tenantKey)], id, true, 'mt-v2');
                    const pointed = pointer?.backend === 'qdrant' && pointer.physical === store.name;
                    // VH-010: an unfinished move may have left vectors here before
                    // writing the saved settings; look whenever one is recorded.
                    if (config.status === 'missing' && !pointed && !looksForPending(ctx, id, 'qdrant', 'mt-v2')) continue;
                    const count = await qdrant.countTenant(store.name, tenantKey);
                    if (count > 0 || config.status !== 'missing') {
                        locations.push({ layout: 'mt-v2', physical: store.name, count, dim: store.dim, config });
                    }
                }
            }
            if (names.has(id)) {
                const info = await qdrantInfo(qdrant, id);
                const count = await qdrantCount(qdrant, id);
                locations.push({ layout: 'per-collection', physical: id, count, dim: info.dim,
                    config: checkRecord(info.metadata[initialConfigKey(id)], id, false) });
            }
            if (legacy.exists) {
                if (legacy.error) {
                    locations.push({ layout: 'legacy-mt', physical: QDRANT_LEGACY_MT, count: null,
                        config: { status: 'unavailable', reason: legacy.error }, error: legacy.error });
                } else {
                    const count = legacy.tally.byType.get(id) || 0;
                    const config = checkRecord(legacy.info.metadata[initialConfigKey(id)], id, true);
                    if (count > 0 || config.status !== 'missing') {
                        locations.push({ layout: 'legacy-mt', physical: QDRANT_LEGACY_MT, count,
                            countNote: legacy.tally.truncated ? 'at least (scan capped)' : undefined,
                            dim: legacy.info.dim, config });
                    }
                }
            }
        } catch (error) {
            row.layout = 'unknown';
            row.error = error.message;
            row.savedConfig = { status: 'unavailable', reason: error.message };
            rows.push(row);
            continue;
        }
        Object.assign(row, deriveRemoteRow(splitOldCopies(row, locations, ctx.oldCopies?.get(id), 'qdrant', ctx)));
        const todayPhysical = ctx.defaults.qdrantMultitenancy ? QDRANT_LEGACY_MT : id;
        // 9c: the legacy settings rule says nothing about a v2 collection.
        row.todayRule = { physical: todayPhysical,
            agrees: row.physical == null || row.layout === 'mt-v2' ? null : row.physical === todayPhysical };
        rows.push(row);
    }

    // Shared stores and stores no record points at.
    const idSet = new Set(ids);
    for (const name of [...names].sort()) {
        if (SHARED_NAME.test(name)) {
            const store = { backend: 'qdrant', name, kind: name === QDRANT_LEGACY_MT ? 'legacy-mt' : MT_V2.test(name) ? 'mt-v2' : 'shared' };
            try {
                if (name === QDRANT_LEGACY_MT && legacy.info) {
                    store.total = legacy.info.points ?? await qdrantCount(qdrant, name);
                    store.dim = legacy.info.dim;
                    if (legacy.tally) {
                        const known = [];
                        let hiddenTypes = 0;
                        let hiddenPoints = 0;
                        for (const [type, count] of [...legacy.tally.byType].sort((a, b) => b[1] - a[1])) {
                            const recorded = idSet.has(type) || ctx.allRecordedIds.has(type);
                            if (recorded || ctx.showForeign) known.push({ contentType: type, count, recorded });
                            else { hiddenTypes++; hiddenPoints += count; }
                        }
                        store.byContentType = known;
                        if (hiddenTypes) store.hidden = { contentTypes: hiddenTypes, points: hiddenPoints };
                        store.withoutContentType = legacy.tally.withoutType;
                        store.scanned = legacy.tally.scanned;
                        store.truncated = legacy.tally.truncated;
                    }
                } else {
                    store.total = await qdrantCount(qdrant, name);
                    store.dim = (await qdrantInfo(qdrant, name)).dim;
                }
            } catch (error) {
                store.error = error.message;
            }
            report.sharedStores.push(store);
        } else if (name.startsWith('vecthare_') && !ctx.allRecordedIds.has(name)) {
            if (!ctx.showForeign) { report.unreferencedHidden.qdrant = (report.unreferencedHidden.qdrant || 0) + 1; continue; }
            const entry = { backend: 'qdrant', name };
            try { entry.count = await qdrantCount(qdrant, name); } catch (error) { entry.error = error.message; }
            report.unreferenced.push(entry);
        }
    }
    return rows;
}

/**
 * 9d: locations named as old copies of a storage move are reported apart
 * ("old copy (moved)") and do not make the collection a conflict. P8: a
 * location stamped with another user's name is flagged.
 */
/** VH-010: does this collection record a pending copy in `layout` whose store is unknown? */
function looksForPending(ctx, id, backend, layout) {
    return (ctx.oldCopies?.get(id) || []).some(copy => copy.pending === true && copy.physical == null &&
        copy.backend === backend && copy.layout === layout);
}

function splitOldCopies(row, locations, oldCopies, backend, ctx) {
    const named = (oldCopies || []).filter(copy => copy.backend === backend);
    const current = [];
    row.oldCopies = [];
    // VH-010: where the collection lives now (its recorded locator); a pending
    // copy whose store is not known yet is any other location in its layout.
    const here = ctx.locators?.get(row.collectionId);
    const herePhysical = here && normBackend(here.backend) === backend ? here.physical : null;
    const matchOf = location => named.find(copy => copy.physical === location.physical) ||
        named.find(copy => copy.pending === true && copy.physical == null && copy.layout === location.layout &&
            location.physical !== herePhysical);
    for (const location of locations) {
        const owner = location.config?.owner;
        if (owner && ctx.userHandle && owner !== ctx.userHandle) location.foreignOwner = owner;
        const copy = matchOf(location);
        if (copy) row.oldCopies.push({ ...location, oldCopy: true, ...(copy.pending ? { pending: true } : {}),
            ...(copy.startedAt ? { startedAt: copy.startedAt } : {}), ...(copy.movedAt ? { movedAt: copy.movedAt } : {}) });
        else current.push(location);
    }
    for (const copy of named) {
        const shown = row.oldCopies.some(l => l.physical === copy.physical ||
            (copy.pending && copy.physical == null && l.pending && l.layout === copy.layout));
        if (!shown) row.oldCopies.push({ ...copy, oldCopy: true, count: 0, note: 'nothing left there' });
    }
    if (!row.oldCopies.length) delete row.oldCopies;
    return current;
}

/** Layout of one remote collection from the locations it was found in. */
function deriveRemoteRow(locations) {
    const layouts = [...new Set(locations.map(l => l.layout))];
    const total = locations.every(l => Number.isSafeInteger(l.count))
        ? locations.reduce((sum, l) => sum + l.count, 0) : null;
    const out = { locations, savedConfig: summarizeConfigs(locations) };
    if (!locations.length) return { ...out, layout: 'not-found', physical: null, count: 0 };
    if (layouts.length > 1) return { ...out, layout: 'conflict', physical: null, count: total };
    const [only] = locations;
    return { ...out, layout: only.layout, physical: only.physical, count: total,
        ...(only.dim ? { dim: only.dim } : {}), ...(only.countNote ? { countNote: only.countNote } : {}) };
}

// ---------------------------------------------------------------------------
// Milvus
// ---------------------------------------------------------------------------

async function milvusSection(ctx, ids, report) {
    const milvus = ctx.milvus;
    const names = new Set(await milvus.listCollectionNames());
    const shared = { exists: names.has(MILVUS_LEGACY_SHARED) };
    if (shared.exists) {
        try {
            shared.described = await milvus.describeProperties(MILVUS_LEGACY_SHARED);
            shared.total = await milvus.countItems(ctx.ALL_TENANTS);
        } catch (error) {
            shared.error = error.message;
        }
    }
    // 9c: v2 stores (vecthare_mt_<dim>) and per-collection stores (vh_*).
    const v2Stores = [];
    for (const name of [...names].filter(n => MT_V2.test(n)).sort()) {
        try { v2Stores.push({ name, ...await milvus.describeProperties(name) }); }
        catch (error) { v2Stores.push({ name, error: error.message }); }
    }
    const ownNames = new Set();

    const rows = [];
    let attributed = 0;
    let attributedComplete = true;
    for (const id of ids) {
        const row = { collectionId: id, backend: 'milvus', todayRule: { physical: MILVUS_LEGACY_SHARED } };
        const v2Locations = [];
        const tenantKey = ctx.tenantKeyOf?.(id);
        if (tenantKey) {
            try {
                const pointer = ctx.locators?.get(id);
                for (const store of v2Stores) {
                    if (store.error) continue;
                    const config = checkRecord(store.properties[initialConfigKeyFor(tenantKey)], id, undefined, 'mt-v2');
                    const pointed = pointer?.backend === 'milvus' && pointer.physical === store.name;
                    if (config.status === 'missing' && !pointed && !looksForPending(ctx, id, 'milvus', 'mt-v2')) continue;
                    const count = await milvus.countV2(store.name, tenantKey);
                    if (count > 0 || config.status !== 'missing') {
                        v2Locations.push({ layout: 'mt-v2', physical: store.name, count, dim: store.dim, config });
                    }
                }
                const own = ctx.milvusOwnName(tenantKey);
                ownNames.add(own);
                if (names.has(own)) {
                    const described = await milvus.describeProperties(own);
                    const count = await milvus.countV2(own, tenantKey);
                    v2Locations.push({ layout: 'per-collection', physical: own, count, dim: described.dim,
                        config: checkRecord(described.properties[initialConfigKeyFor(tenantKey)], id, undefined, 'per-collection') });
                }
            } catch (error) {
                v2Locations.push({ layout: 'unknown', physical: null, count: null, config: { status: 'unavailable', reason: error.message } });
            }
        }
        const tenant = ctx.parseMilvusTenant(id);
        if (!tenant) {
            rows.push({ ...row, layout: 'unknown', physical: null, count: null,
                savedConfig: { status: 'unavailable', reason: 'id cannot be mapped to a Milvus collection identity' },
                notes: ['This id cannot be addressed in the shared Milvus store.'] });
            continue;
        }
        const locations = [];
        try {
            if (shared.exists) {
                if (shared.error) throw new Error(shared.error);
                const count = await milvus.countItems(tenant);
                attributed += count;
                const config = checkRecord(shared.described.properties[initialConfigKey(id)], id);
                if (count > 0 || config.status !== 'missing') {
                    locations.push({ layout: 'legacy-shared', physical: MILVUS_LEGACY_SHARED, count,
                        dim: shared.described.dim, config });
                }
            }
        } catch (error) {
            attributedComplete = false;
            rows.push({ ...row, layout: 'unknown', error: error.message,
                savedConfig: { status: 'unavailable', reason: error.message } });
            continue;
        }
        Object.assign(row, deriveRemoteRow(splitOldCopies(row, [...v2Locations, ...locations], ctx.oldCopies?.get(id), 'milvus', ctx)));
        row.todayRule.agrees = row.physical == null || row.layout !== 'legacy-shared' ? null : row.physical === MILVUS_LEGACY_SHARED;
        rows.push(row);
    }
    for (const name of [...names].sort()) {
        if (SHARED_NAME.test(name)) {
            const store = { backend: 'milvus', name, kind: name === MILVUS_LEGACY_SHARED ? 'legacy-shared' : 'shared' };
            const v2 = v2Stores.find(x => x.name === name);
            if (v2) {
                store.kind = 'mt-v2';
                store.dim = v2.dim ?? null;
                if (v2.error) store.error = v2.error;
            }
            if (name === MILVUS_LEGACY_SHARED) {
                if (shared.error) store.error = shared.error;
                else {
                    store.total = shared.total;
                    store.dim = shared.described?.dim ?? null;
                    if (attributedComplete) {
                        store.attributedToYourRecords = attributed;
                        store.notAttributed = Math.max(0, shared.total - attributed);
                    }
                }
            }
            report.sharedStores.push(store);
        } else if ((name.startsWith('vecthare_') || name.startsWith('vh_')) && !ctx.allRecordedIds.has(name) && !ownNames.has(name)) {
            if (!ctx.showForeign) { report.unreferencedHidden.milvus = (report.unreferencedHidden.milvus || 0) + 1; continue; }
            report.unreferenced.push({ backend: 'milvus', name });
        }
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Vectra / Standard (per-user folders)
// ---------------------------------------------------------------------------

async function listDirs(fs, dir) {
    try {
        return (await fs.readdir(dir, { withFileTypes: true }))
            .filter(e => e.isDirectory() && !e.isSymbolicLink()).map(e => e.name);
    } catch (error) {
        if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return [];
        throw error;
    }
}

async function readJsonFile(fs, file, maxBytes) {
    let stat;
    try { stat = await fs.lstat(file); } catch (error) { if (error.code === 'ENOENT') return { status: 'absent' }; throw error; }
    if (!stat.isFile() || stat.isSymbolicLink()) return { status: 'unreadable' };
    if (stat.size > maxBytes) return { status: 'too-large', size: stat.size };
    try { return { status: 'ok', value: JSON.parse(await fs.readFile(file, 'utf8')) }; }
    catch { return { status: 'unreadable' }; }
}

/** Every VectHare-named Vectra index folder in this user's vectors directory. */
async function scanVectra(ctx) {
    const { fs, vectorsDir } = ctx;
    const byFolder = new Map(); // collection folder name -> [{source, model, physical, count, config}]
    for (const source of await listDirs(fs, vectorsDir)) {
        if (source === 'lancedb' || source.startsWith('vecthare-owned')) continue;
        for (const folder of await listDirs(fs, path.join(vectorsDir, source))) {
            if (!folder.startsWith('vecthare_')) continue;
            const base = path.join(vectorsDir, source, folder);
            const candidates = [{ model: '', dir: base }];
            for (const child of await listDirs(fs, base)) candidates.push({ model: child, dir: path.join(base, child) });
            for (const { model, dir } of candidates) {
                const index = await readJsonFile(fs, path.join(dir, 'index.json'), MAX_VECTRA_INDEX_BYTES);
                if (index.status === 'absent') continue;
                const entry = { source, model, physical: [source, folder, ...(model ? [model] : [])].join('/') };
                entry.count = index.status === 'ok' && Array.isArray(index.value?.items) ? index.value.items.length : null;
                if (entry.count === null) entry.countNote = index.status === 'too-large' ? 'index too large to count here' : 'index unreadable';
                const cfg = await readJsonFile(fs, path.join(dir, 'vecthare-initial-config.json'), 65536);
                entry.configRaw = cfg;
                const list = byFolder.get(folder) || [];
                list.push(entry);
                byFolder.set(folder, list);
            }
        }
    }
    return byFolder;
}

function localConfig(entry, id) {
    const cfg = entry.configRaw;
    if (cfg.status === 'absent') return { status: 'missing' };
    if (cfg.status !== 'ok') return { status: 'invalid', reason: 'unreadable' };
    const checked = checkRecord(cfg.value, id);
    if (checked.status === 'found' && checked.provider !== entry.source) {
        return { status: 'invalid', reason: 'records another embedding source than its folder' };
    }
    return checked;
}

function localRows(backend, ids, byName, nameOf, defaults) {
    const rows = [];
    for (const id of ids) {
        const found = (byName.get(nameOf(id)) || []).map(entry => ({
            layout: 'per-collection', physical: entry.physical, source: entry.source,
            ...(entry.model ? { model: entry.model } : {}),
            count: entry.count, ...(entry.countNote ? { countNote: entry.countNote } : {}),
            config: localConfig(entry, id),
        }));
        const count = found.length && found.every(l => Number.isSafeInteger(l.count))
            ? found.reduce((sum, l) => sum + l.count, 0) : (found.length ? null : 0);
        const row = {
            collectionId: id, backend,
            layout: found.length ? 'per-collection' : 'not-found',
            physical: found.length === 1 ? found[0].physical : null,
            count, locations: found, savedConfig: summarizeConfigs(found),
            todayRule: { source: defaults.source || null,
                agrees: found.length ? found.some(l => l.source === defaults.source) : null },
        };
        if (found.length > 1) row.notes = [`${found.length} index folders hold this collection (several providers or models).`];
        rows.push(row);
    }
    return rows;
}

async function scanLance(ctx) {
    const { fs, vectorsDir } = ctx;
    const root = path.join(vectorsDir, 'lancedb');
    const sources = await listDirs(fs, root);
    const byTable = new Map();
    if (!sources.length) return byTable;
    const lance = await ctx.lance();
    for (const source of sources) {
        if (!/^[A-Za-z0-9_-]+$/.test(source)) continue;
        let tables;
        try { tables = await (await lance.getDatabase(source)).tableNames(); }
        catch (error) { byTable.set(`\u0000error:${source}`, [{ source, error: error.message }]); continue; }
        for (const table of tables || []) {
            if (typeof table !== 'string' || !table.startsWith('vecthare_')) continue;
            const entry = { source, physical: `lancedb/${source}/${table}` };
            try {
                const stats = await lance.getCollectionStats(table, source);
                const count = typeof stats?.count === 'bigint' ? Number(stats.count) : stats?.count;
                entry.count = Number.isSafeInteger(count) ? count : null;
                if (Number.isSafeInteger(stats?.dimension) && stats.dimension > 0) entry.dim = stats.dimension;
            } catch (error) {
                entry.count = null;
                entry.countNote = error.message;
            }
            entry.configRaw = safeToken(table)
                ? await readJsonFile(fs, path.join(root, source, `${table}.lance`, 'vecthare-initial-config.json'), 65536)
                : { status: 'absent' };
            const list = byTable.get(table) || [];
            list.push(entry);
            byTable.set(table, list);
        }
    }
    return byTable;
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/**
 * @param {object} ctx
 * @param {string} ctx.vectorsDir              this user's vectors directory
 * @param {string} ctx.dataRoot                SillyTavern data root (user folders)
 * @param {boolean} ctx.isAdmin                caller is a SillyTavern admin
 * @param {object} ctx.defaults                {backend, source, model, qdrantMultitenancy}: today's settings
 * @param {Function} ctx.readOwnedState        () => {status, state?, token}
 * @param {object|null} ctx.qdrant             connected Qdrant backend, or null
 * @param {object|null} ctx.milvus             connected Milvus backend, or null
 * @param {Function} ctx.lance                 () => LanceDB backend for this user
 * @param {Function} ctx.sanitize              sanitize-filename
 * @param {Function} ctx.parseMilvusTenant
 * @param {object} ctx.ALL_TENANTS
 */
export async function buildStorageReport(ctx) {
    ctx.fs ??= fsDefault;
    const defaults = {
        backend: normBackend(KNOWN_BACKENDS.includes(ctx.defaults?.backend) ? ctx.defaults.backend : 'standard'),
        source: safeToken(ctx.defaults?.source) ? ctx.defaults.source : null,
        model: typeof ctx.defaults?.model === 'string' ? ctx.defaults.model.slice(0, 256) : '',
        qdrantMultitenancy: ctx.defaults?.qdrantMultitenancy === true,
    };
    ctx.defaults = defaults;
    const server = await countServerUsers(ctx.dataRoot, ctx.fs);
    // Other users' names are only shown to an admin, or on a confirmed
    // single-user server.
    ctx.showForeign = ctx.isAdmin === true || (server.confirmed && server.mode === 'single-user');

    const report = {
        success: true,
        version: STORAGE_REPORT_VERSION,
        generatedAt: new Date().toISOString(),
        readOnly: true,
        server: { ...server, admin: ctx.isAdmin === true },
        defaults,
        state: null,
        backends: {},
        collections: [],
        sharedStores: [],
        unreferenced: [],
        unreferencedHidden: {},
        warnings: [],
    };

    const owned = await ctx.readOwnedState();
    report.state = { status: owned.status, ...(owned.token ? { token: owned.token } : {}) };
    const recorded = owned.status === 'found' ? collectRecordedCollections(owned.state) : new Map();
    if (recorded.size > MAX_IDS) {
        report.warnings.push(`Only the first ${MAX_IDS} of ${recorded.size} recorded collections are checked.`);
    }
    const entries = [...recorded].slice(0, MAX_IDS);
    ctx.allRecordedIds = new Set(recorded.keys());
    ctx.locators = new Map([...recorded].map(([id, e]) => [id, summarizeLocator(e.locator)]).filter(([, l]) => l));
    ctx.oldCopies = new Map([...recorded].filter(([, e]) => e.oldCopies.length).map(([id, e]) => [id, e.oldCopies]));
    report.state.records = recorded.size;

    // Which backend(s) each collection is checked on: what its records say,
    // or today's backend for a record that names none.
    const perBackend = new Map(KNOWN_BACKENDS.filter(b => b !== 'vectra').map(b => [b, []]));
    const hintsById = new Map();
    for (const [id, entry] of entries) {
        const backends = entry.backends.size ? [...entry.backends.keys()] : [defaults.backend];
        hintsById.set(id, { hints: [...entry.hints].sort(), assumedBackend: entry.backends.size === 0,
            sources: Object.fromEntries([...entry.backends].map(([b, s]) => [b, [...s].sort()])),
            locator: summarizeLocator(entry.locator) });
        // A recorded locator names the backend too (Step 9b).
        const located = normBackend(entry.locator?.backend);
        if (entry.locator && KNOWN_BACKENDS.includes(located) && !entry.backends.has(located)) backends.push(located);
        for (const b of new Set(backends.map(normBackend))) perBackend.get(b)?.push(id);
    }

    const rows = [];
    const remote = { qdrant: ctx.qdrant, milvus: ctx.milvus };
    for (const backend of REMOTE_BACKENDS) {
        const ids = perBackend.get(backend);
        const client = remote[backend];
        report.backends[backend] = { connected: Boolean(client), records: ids.length };
        if (!client) {
            for (const id of ids) {
                rows.push({ collectionId: id, backend, layout: 'unknown', physical: null, count: null,
                    savedConfig: { status: 'unavailable', reason: `${backend === 'qdrant' ? 'Qdrant' : 'Milvus'} is not connected on the server` },
                    notes: [`Select ${backend === 'qdrant' ? 'Qdrant' : 'Milvus'} in VectHare once so the server connects, then run the report again.`] });
            }
            continue;
        }
        try {
            rows.push(...(backend === 'qdrant'
                ? await qdrantSection(ctx, ids, report)
                : await milvusSection(ctx, ids, report)));
        } catch (error) {
            report.backends[backend].error = error.message;
            for (const id of ids) {
                rows.push({ collectionId: id, backend, layout: 'unknown', physical: null, count: null, error: error.message,
                    savedConfig: { status: 'unavailable', reason: error.message } });
            }
        }
    }

    // Standard / Vectra
    {
        const ids = perBackend.get('standard');
        report.backends.standard = { connected: true, records: ids.length };
        try {
            const byFolder = await scanVectra(ctx);
            rows.push(...localRows('standard', ids, byFolder, id => ctx.sanitize(id), defaults));
            const wanted = new Set([...ctx.allRecordedIds].map(id => ctx.sanitize(id)));
            for (const [folder, list] of byFolder) {
                if (wanted.has(folder)) continue;
                for (const entry of list) {
                    report.unreferenced.push({ backend: 'standard', name: entry.physical, count: entry.count });
                }
            }
        } catch (error) {
            report.backends.standard.error = error.message;
        }
    }

    // LanceDB
    {
        const ids = perBackend.get('lancedb');
        const tableOf = id => id.replace(/[^a-zA-Z0-9_.-]/g, '_');
        report.backends.lancedb = { connected: true, records: ids.length };
        try {
            const byTable = await scanLance(ctx);
            for (const [key, list] of byTable) {
                if (key.startsWith('\u0000error:')) report.warnings.push(`LanceDB source ${list[0].source}: ${list[0].error}`);
            }
            rows.push(...localRows('lancedb', ids, byTable, tableOf, defaults));
            const wanted = new Set([...ctx.allRecordedIds].map(tableOf));
            for (const [table, list] of byTable) {
                if (table.startsWith('\u0000') || wanted.has(table)) continue;
                for (const entry of list) report.unreferenced.push({ backend: 'lancedb', name: entry.physical, count: entry.count });
            }
        } catch (error) {
            report.backends.lancedb.error = error.message;
        }
    }

    for (const row of rows) {
        const { locator, ...hints } = hintsById.get(row.collectionId) || {};
        Object.assign(row, hints);
        if (locator && locator.backend === row.backend) {
            row.locator = locator;
            row.locatorAgrees = row.physical != null && locator.valid && locator.physical === row.physical;
        }
        // Internal raw config is never sent to the browser.
        for (const location of row.locations || []) delete location.configRaw;
    }
    rows.sort((a, b) => a.collectionId.localeCompare(b.collectionId) || a.backend.localeCompare(b.backend));
    report.collections = rows;

    const tally = {};
    for (const row of rows) tally[row.layout] = (tally[row.layout] || 0) + 1;
    report.summary = {
        collections: rows.length,
        layouts: tally,
        configMissing: rows.filter(r => r.savedConfig?.status === 'missing').length,
        configMultiple: rows.filter(r => r.savedConfig?.status === 'multiple').length,
        configInvalid: rows.filter(r => r.savedConfig?.status === 'invalid').length,
        todayRuleDisagrees: rows.filter(r => r.todayRule?.agrees === false).length,
        locators: rows.filter(r => r.locator).length,
        locatorDisagrees: rows.filter(r => r.locator && !r.locatorAgrees).length,
        unreferenced: report.unreferenced.length +
            Object.values(report.unreferencedHidden).reduce((a, b) => a + b, 0),
    };
    return report;
}
