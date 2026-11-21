/**
 * Similharity Server Plugin
 *
 * Unified vector database backend for VectHare extension.
 * Supports multiple backends: Vectra (file-based), LanceDB, Qdrant
 *
 * All chunk operations go through unified /chunks/* endpoints.
 * Backend is specified via `backend` parameter in request body.
 *
 * @version see package.json
 */

import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import dns from 'node:dns/promises';
import net from 'node:net';
import sanitize from 'sanitize-filename';
import vectra from 'vectra';
import { LanceDBBackend } from './lancedb-backend.js';
import qdrantBackend, { isLegacyContentType, legacyContentTypeFromFilter } from './qdrant-backend.js';
import { buildStorageReport, collectRecordedCollections } from './storage-report.js';
import {
    V2_CAPABILITY, MT_V2_NAME, tenantFor, configKeyForTenant, parseV2Storage, v2PhysicalName,
    milvusOwnName, mtV2Name, userHandle,
} from './tenancy.js';
import { copyCollection, deleteCopy, parseLocation } from './storage-move.js';
import { isReservedSharedCollection } from './qdrant-backend.js';
import milvusBackend, { ALL_TENANTS, parseMilvusTenant, normalizeMilvusTenant } from './milvus-backend.js';
import { toSafeHashList, requireSafeHash, isSafeFilterString } from './safe-values.js';

const pluginName = 'similharity';
// One source for the version: package.json. VectHare shows it beside its own
// version; plugin and extension are released with the same version number.
const pluginVersion = (() => {
    try { return String(JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')).version || '0.0.0'); } catch { return '0.0.0'; }
})();
// Compatibility number for the extension <-> plugin contract (routes, request
// and answer shapes, safety rules). Raise it ONLY when that contract changes in
// a way an older extension or plugin cannot handle; VectHare declares the
// number it needs (core/plugin-health.js PLUGIN_PROTOCOL) and shows a mismatch.
const PLUGIN_PROTOCOL = 1;

// Items in a plain Vectra index folder, or null when the folder holds anything
// other than index.json (and VectHare's own config file) or cannot be parsed.
// Used to prove a model folder is empty before a whole-collection purge.
async function countPlainVectraIndexItems(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const allowed = new Set(['index.json', 'vecthare-initial-config.json']);
    if (entries.some(entry => !entry.isFile() || entry.isSymbolicLink() || !allowed.has(entry.name)) ||
        !entries.some(entry => entry.name === 'index.json')) return null;
    try {
        const index = JSON.parse(await fs.readFile(path.join(dir, 'index.json'), 'utf8'));
        return Array.isArray(index?.items) ? index.items.length : null;
    } catch {
        return null;
    }
}

// LanceDB keeps its base path and open-table caches on the backend object.
// A single shared object bound itself to whichever SillyTavern user called
// first, so later users read, wrote and deleted inside that first user's
// folder. One instance per vectors directory keeps every user in their own.
const lanceInstances = new Map();

/**
 * The initialised LanceDB backend for one user's vectors directory.
 * @param {{vectors: string}} directories req.user.directories
 * @returns {Promise<LanceDBBackend>}
 */
async function lanceFor(directories) {
    if (!directories || typeof directories.vectors !== 'string' || !directories.vectors) {
        throw new Error('No vectors directory for this user');
    }
    const key = path.resolve(directories.vectors);
    let entry = lanceInstances.get(key);
    if (!entry) {
        const instance = new LanceDBBackend();
        entry = { instance, ready: instance.initialize(key) };
        // A failed initialisation must not stay cached forever.
        entry.ready.catch(() => { if (lanceInstances.get(key) === entry) lanceInstances.delete(key); });
        lanceInstances.set(key, entry);
    }
    await entry.ready;
    return entry.instance;
}

/**
 * Initialize the plugin
 * @param {import('express').Router} router - Express router for plugin endpoints
 */
export async function init(router) {
    console.log(`[${pluginName}] Initializing v${pluginVersion}...`);

    // ========================================================================
    // BACKEND HANDLER - Routes requests to appropriate backend
    // ========================================================================

    /**
     * Ensures LanceDB is initialized before use
     * @param {object} directories - User directories containing vectors path
     */
    async function ensureLanceDBInitialized(directories) {
        await lanceFor(directories);
    }

    // This plugin may share SillyTavern's vector directory. A caller must not
    // be able to write, update or delete a vanilla collection through VectHare.
    const isVectHareCollection = id => typeof id === 'string' &&
        /^vecthare_[A-Za-z0-9_.-]+$/.test(id);
    function requireVectHareCollection(req, res, next) {
        if (!isVectHareCollection(req.body?.collectionId)) {
            return res.status(403).json({ error: 'Only VectHare-owned collection IDs may be modified' });
        }
        next();
    }

    // ------------------------------------------------------------------------
    // Request guards for /chunks/*
    //
    // Milvus keeps every VectHare collection in ONE physical collection and the
    // extension always addresses it as 'vecthare_main'. Which logical collection
    // a call belongs to therefore arrives separately. The scope is taken from
    // the logical collection id (`tenantCollectionId`, parsed exactly like the
    // extension does) or, for older extension builds, from strictly validated
    // `filters.type` / `filters.sourceId`. Either way the plugin REPLACES the
    // client's filters with the validated scope, and refuses the call when no
    // valid scope can be established: the unscoped form of these calls is what
    // let one chat's delete reach another chat's vectors.
    // ------------------------------------------------------------------------
    const CHUNK_STATIC_PATHS = new Set(['list', 'insert', 'delete', 'query', 'hybrid-query', 'purge', 'stats', 'get-by-hashes']);

    function resolveMilvusScope(source) {
        const clientFilters = source?.filters && typeof source.filters === 'object' ? source.filters : {};
        const hasTypeOrSource = clientFilters.type !== undefined || clientFilters.sourceId !== undefined;
        let tenant = null;
        if (source?.tenantCollectionId !== undefined) {
            tenant = parseMilvusTenant(source.tenantCollectionId);
            if (!tenant) return { error: 'tenantCollectionId is not a valid VectHare collection id' };
            if (hasTypeOrSource && (clientFilters.type !== tenant.type || clientFilters.sourceId !== tenant.sourceId)) {
                return { error: 'filters do not match tenantCollectionId' };
            }
        } else {
            tenant = normalizeMilvusTenant({ type: clientFilters.type, sourceId: clientFilters.sourceId });
            if (!tenant || tenant === ALL_TENANTS) {
                return { error: 'Milvus requests must identify their collection (tenantCollectionId or filters.type + filters.sourceId)' };
            }
        }
        const scoped = { type: tenant.type, sourceId: tenant.sourceId };
        // Optional read-only refinements survive only if they validate.
        if (clientFilters.embeddingSource !== undefined && clientFilters.embeddingSource !== '') {
            if (!isSafeFilterString(clientFilters.embeddingSource, 64)) return { error: 'Invalid embeddingSource filter' };
            scoped.embeddingSource = clientFilters.embeddingSource;
        }
        if (clientFilters.timestampAfter !== undefined && clientFilters.timestampAfter !== null && clientFilters.timestampAfter !== '') {
            if (!Number.isSafeInteger(Number(clientFilters.timestampAfter))) return { error: 'Invalid timestampAfter filter' };
            scoped.timestampAfter = Number(clientFilters.timestampAfter);
        }
        return { scoped };
    }

    // VH-054: after SillyTavern restarts, the shared Qdrant/Milvus clients are
    // empty until someone calls /backend/init. A request that needs one brings
    // back the last working connection (remembered at init, keys from the key
    // store) instead of failing with "not initialized" until the page reloads.
    // VH-094: once per user (first request after the plugin starts), move
    // Standard collections that older builds wrote to the folder without a
    // model into SillyTavern's own folder, before anything reads them through
    // SillyTavern's /api/vector/* (which only looks there).
    const vectraScopeScans = new Map();
    router.use(async (req, res, next) => {
        const directories = req.user?.directories;
        const handle = req.user?.profile?.handle || directories?.vectors;
        if (directories?.vectors && handle) {
            if (!vectraScopeScans.has(handle)) {
                vectraScopeScans.set(handle, moveOldVectraScopes(directories).catch((error) => {
                    vectraScopeScans.delete(handle);
                    console.warn(`[${pluginName}] Standard collection folder check skipped: ${error?.message || error}`);
                }));
            }
            await vectraScopeScans.get(handle);
        }
        next();
    });

    const restoring = new Map();
    const LISTS_EVERY_BACKEND = new Set(['/collections', '/storage/report', '/collections/paths']);
    router.use(async (req, res, next) => {
        try {
            if (req.path.startsWith('/backend/init/')) return next();
            const named = req.body?.backend ?? req.query?.backend;
            const wanted = LISTS_EVERY_BACKEND.has(req.path) ? ['qdrant', 'milvus']
                : (named === 'qdrant' || named === 'milvus') ? [named] : [];
            for (const backend of wanted) {
                if (isBackendReady(backend)) continue;
                let pending = restoring.get(backend);
                if (!pending) {
                    pending = restoreServerConnection(req, backend)
                        .catch(error => { console.warn(`[${pluginName}] Could not restore the ${backend} connection: ${error.message}`); return false; })
                        .finally(() => restoring.delete(backend));
                    restoring.set(backend, pending);
                }
                await pending;
            }
        } catch (error) {
            console.warn(`[${pluginName}] Connection restore skipped: ${error.message}`);
        }
        next();
    });

    router.use('/chunks', async (req, res, next) => {
        try {
            const isGet = req.method === 'GET';
            const source = isGet ? req.query : req.body;
            const backend = source?.backend;

            // Hash values and hash lists are interpolated into Milvus and
            // LanceDB filters: accept only safe integers, reject otherwise.
            if (backend === 'milvus' || backend === 'lancedb') {
                const [first] = req.path.split('/').filter(Boolean);
                if (first && !CHUNK_STATIC_PATHS.has(first)) {
                    let hash = first;
                    try { hash = decodeURIComponent(first); } catch { /* validated below */ }
                    requireSafeHash(hash);
                }
                if (!isGet && req.body?.hashes !== undefined) toSafeHashList(req.body.hashes);
            }

            // VH-015: on Qdrant older than 1.15 no collection can keep its saved
            // settings; refuse writes up front instead of half-writing a collection.
            if (backend === 'qdrant' && !isGet && req.path === '/insert' && qdrantBackend.supportsCollectionMetadata() === false) {
                return res.status(409).json({ error: qdrantBackend.tooOldMessage(), code: 'qdrant_too_old' });
            }

            // 9d (P8): a per-collection Qdrant store stamped with another user's
            // name is refused on a multi-user server (names are global there).
            if (backend === 'qdrant' && (storageRaw0(req, isGet) === undefined || storageRaw0(req, isGet) === null)) {
                const denied = await qdrantOwnerDenied(req, source?.collectionId);
                if (denied) return res.status(403).json({ error: denied });
                // VH-004: remember who creates a new per-collection store, so it is theirs
                // until its saved settings carry the owner stamp.
                if (!isGet && req.path === '/insert' && typeof source?.collectionId === 'string' && qdrantBackend.baseUrl &&
                    req.user?.profile?.admin !== true && await isMultiUserServer(req)) {
                    const name = qdrantBackend._parseCollectionName(source.collectionId);
                    if (!isReservedSharedCollection(name) && !await qdrantBackend.collectionExists(name).catch(() => true)) {
                        await rememberCreator(req, `qdrant:${name}`);
                    }
                }
                // VH-033: the old shared store has no user scope; reads by id (GET) cannot be scoped.
                if (isGet && typeof source?.collectionId === 'string' &&
                    qdrantBackend._parseCollectionName(source.collectionId) === 'vecthare_multitenancy' &&
                    req.user?.profile?.admin !== true && await isMultiUserServer(req)) {
                    return res.status(403).json({ error: 'On a multi-user server the older shared Qdrant store can only be read for one named collection' });
                }
            }

            // 9c: a request naming a v2 storage layout (Qdrant/Milvus mt-v2,
            // Milvus per-collection) is scoped HERE: the tenant comes from the
            // SillyTavern user and the validated logical id, client filters are
            // dropped, and the physical store is resolved. Nothing else applies.
            let storageRaw = isGet ? req.query?.storage : req.body?.storage;
            if (isGet && typeof storageRaw === 'string') {
                try { storageRaw = JSON.parse(storageRaw); }
                catch { return res.status(400).json({ error: 'storage must be valid JSON' }); }
            }
            if (storageRaw !== undefined && storageRaw !== null) {
                const resolved = await resolveV2Request(req, backend, source?.collectionId, storageRaw);
                if (resolved.error) return res.status(resolved.status || 400).json({ error: resolved.error });
                req.vh2 = resolved;
                if (isGet) req.vecthareMilvusFilters = undefined;
                else req.body.filters = {};
                return next();
            }

            // 9a: in the legacy shared Qdrant store a delete must be limited to
            // the caller's own points. The extension sends that scope as
            // `filter` (singular), which used to be ignored; it is carried into
            // `filters.content_type` for deletes. 9c: a caller can ask for scoped
            // reads by naming `filters.content_type` explicitly; reads without it
            // stay unscoped (older extension builds rely on that).
            if (backend === 'qdrant' && !isGet && typeof req.body?.collectionId === 'string' &&
                qdrantBackend._parseCollectionName(req.body.collectionId) === 'vecthare_multitenancy') {
                const given = req.body.filters?.content_type;
                const fromFilter = legacyContentTypeFromFilter(req.body.filter);
                if (given !== undefined && !isLegacyContentType(given)) {
                    return res.status(400).json({ error: 'Invalid content_type filter' });
                }
                if (given !== undefined && fromFilter && given !== fromFilter) {
                    return res.status(400).json({ error: 'filter and filters name different collections' });
                }
                // VH-033: on a multi-user server a non-admin must name the collection
                // (any request, not only deletes), and must be its owner by evidence.
                const restricted = req.user?.profile?.admin !== true && await isMultiUserServer(req);
                const scope = given ?? ((req.path === '/delete' || restricted) ? fromFilter : null);
                if (restricted) {
                    const denied = await legacyOwnerDenied(req, scope || null);
                    if (denied && !(scope && req.path !== '/insert' && await legacyMtScopeEmpty(scope))) {
                        return res.status(403).json({ error: denied });
                    }
                }
                if (scope) {
                    const base = req.body.filters && typeof req.body.filters === 'object' && !Array.isArray(req.body.filters)
                        ? req.body.filters : {};
                    req.body.filters = { ...base, content_type: scope };
                }
            }

            if (backend === 'milvus') {
                let raw = source;
                if (isGet && typeof req.query.filters === 'string') {
                    try { raw = { ...req.query, filters: JSON.parse(req.query.filters) }; }
                    catch { return res.status(400).json({ error: 'filters must be valid JSON' }); }
                }
                const { scoped, error } = resolveMilvusScope(raw);
                if (error) return res.status(400).json({ error });
                // VH-002: vecthare_main rows carry no user. Non-admins on a multi-user
                // server need evidence; a tenant they create here is remembered as theirs.
                if (req.user?.profile?.admin !== true && await isMultiUserServer(req)) {
                    const logical = `vecthare_${scoped.type}_${scoped.sourceId}`;
                    const creatorKey = `milvus:${logical}`;
                    if (!isGet && req.path === '/insert' && milvusBackend.isConnected &&
                        await milvusBackend.countItems({ type: scoped.type, sourceId: scoped.sourceId }).catch(() => 1) === 0) {
                        await rememberCreator(req, creatorKey);
                    }
                    let denied = await legacyOwnerDenied(req, logical, creatorKey);
                    const asSent = typeof raw?.tenantCollectionId === 'string' ? raw.tenantCollectionId : null;
                    if (denied && asSent && asSent !== logical && !await legacyOwnerDenied(req, asSent, creatorKey)) denied = null;
                    if (denied && req.path !== '/insert' &&
                        await milvusBackend.countItems({ type: scoped.type, sourceId: scoped.sourceId }).catch(error => (/not exist|not found|not initialized/i.test(error?.message || '') ? 0 : 1)) === 0) {
                        denied = null; // nothing stored for this tenant: nothing to protect
                    }
                    if (denied) return res.status(403).json({ error: denied });
                }
                // Everything downstream sees only the validated scope.
                if (isGet) req.vecthareMilvusFilters = scoped;
                else req.body.filters = scoped;
            }
            next();
        } catch (error) {
            res.status(error.status === 400 ? 400 : 500).json({ error: error.message });
        }
    });

    const storageRaw0 = (req, isGet) => isGet ? req.query?.storage : req.body?.storage;

    // 9d (P8): owner of a per-collection Qdrant store (its saved settings).
    // A stamp is cached for a minute; "no stamp" only for a few seconds, since
    // a new collection gets its stamp moments after its first insert.
    // undefined = no stamp (older collection).
    const ownerCache = new Map();
    const ownerCacheKey = name => `${qdrantBackend.baseUrl}\u0000${name}`;
    async function qdrantOwnerOf(name) {
        const key = ownerCacheKey(name);
        const hit = ownerCache.get(key);
        if (hit && hit.at > Date.now() - (hit.owner === undefined ? 5_000 : 60_000)) return hit.owner;
        // VH-003: a failed read is not "no stamp": it throws (and is not cached),
        // so the guard fails closed instead of open.
        const record = await qdrantBackend.readInitialCollectionConfig(name,
            `vecthare_initial_${createHash('sha256').update(name).digest('hex')}`);
        const owner = record && typeof record.owner === 'string' ? record.owner : undefined;
        ownerCache.set(key, { owner, at: Date.now() });
        return owner;
    }
    // ------------------------------------------------------------------------
    // Decision B (VH-002, VH-004, VH-033): stores from before per-user tenancy
    // (unstamped per-collection Qdrant stores, the shared Qdrant
    // vecthare_multitenancy store and Milvus vecthare_main) carry no user. On a
    // multi-user server a non-admin may use such a collection only when the
    // evidence names them: exactly one user's own collections.json lists the
    // logical id, and it is theirs. Nobody or several -> admin only. A store a
    // user creates through this plugin is remembered as theirs until its saved
    // settings carry the owner stamp. Single-user servers are unaffected.
    // ------------------------------------------------------------------------
    const evidenceFileCache = new Map(); // filename -> { mtimeMs, size, ids }
    async function idsListedIn(filename) {
        let stat;
        try { stat = await fs.stat(filename); } catch { return null; }
        const hit = evidenceFileCache.get(filename);
        if (hit && hit.mtimeMs === stat.mtimeMs && hit.size === stat.size) return hit.ids;
        const current = await readOwnedStateFile(filename).catch(() => null);
        if (current?.status !== 'found') { evidenceFileCache.delete(filename); return null; }
        const ids = new Set();
        const add = key => {
            if (typeof key !== 'string') return;
            ids.add(key);
            const qualified = /^(?:[A-Za-z0-9_.-]+:){1,2}(vecthare_[A-Za-z0-9_.-]+)$/.exec(key);
            if (qualified) ids.add(qualified[1]);
        };
        current.state.registry.forEach(add);
        Object.keys(current.state.collections).forEach(add);
        evidenceFileCache.set(filename, { mtimeMs: stat.mtimeMs, size: stat.size, ids });
        return ids;
    }
    const dataRootOf = req => path.dirname(req.user.directories.root ?? path.dirname(req.user.directories.vectors));
    async function evidenceOwners(req, logicalId) {
        const owners = [];
        let entries = [];
        try { entries = await fs.readdir(dataRootOf(req), { withFileTypes: true }); } catch { return owners; }
        for (const entry of entries) {
            if (!entry.isDirectory() || entry.name.startsWith('_') || entry.name.startsWith('.')) continue;
            const ids = await idsListedIn(path.join(dataRootOf(req), entry.name, 'vectors', OWNED_STATE_DIRECTORY, OWNED_STATE_FILE));
            if (ids?.has(logicalId)) owners.push(entry.name);
        }
        return owners;
    }

    // Stores created through this plugin before their owner stamp exists.
    const creatorsFile = req => path.join(dataRootOf(req), '_vecthare-server', 'store-creators.json');
    let creatorsCache = null;
    async function readCreators(req) {
        if (creatorsCache) return creatorsCache;
        try {
            const parsed = JSON.parse(await fs.readFile(creatorsFile(req), 'utf8'));
            creatorsCache = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
        } catch { creatorsCache = {}; }
        return creatorsCache;
    }
    let creatorsWrite = Promise.resolve();
    async function rememberCreator(req, key) {
        const all = await readCreators(req);
        if (all[key]) return;
        all[key] = { by: userHandle(req), at: new Date().toISOString() };
        const file = creatorsFile(req);
        creatorsWrite = creatorsWrite.then(async () => {
            await fs.mkdir(path.dirname(file), { recursive: true });
            const temporary = `${file}.${randomUUID()}.tmp`;
            await fs.writeFile(temporary, JSON.stringify(all, null, 2), { encoding: 'utf8', mode: 0o600 });
            await fs.rename(temporary, file);
        }).catch(error => console.warn(`[${pluginName}] Could not record a new store's creator: ${error.message}`));
        await creatorsWrite;
    }
    const createdBy = async (req, key) => (await readCreators(req))[key]?.by;

    // A collection with no rows (and no saved settings) in an old shared store has
    // nothing to protect: probes for it (the locator, purge of all locations) may
    // see "empty" instead of a refusal.
    async function legacyMtScopeEmpty(contentType) {
        try {
            const r = await qdrantBackend._request('POST', '/collections/vecthare_multitenancy/points/count',
                { exact: true, filter: { must: [{ key: 'content_type', match: { value: contentType } }] } });
            return r?.result?.count === 0;
        } catch (error) {
            return error?.status === 404; // the shared store does not exist
        }
    }

    /** @returns {Promise<string|null>} a refusal message, or null when allowed */
    async function legacyOwnerDenied(req, logicalId, creatorKey = null) {
        if (req.user?.profile?.admin === true || !await isMultiUserServer(req)) return null;
        if (typeof logicalId !== 'string' || !logicalId) {
            return 'On a multi-user server this older shared store can only be used for one named collection';
        }
        const owners = await evidenceOwners(req, logicalId);
        if (owners.length === 1 && owners[0] === userHandle(req)) return null;
        if (owners.length === 0 && creatorKey && await createdBy(req, creatorKey) === userHandle(req)) return null;
        return owners.length > 1
            ? 'This older collection is listed by several SillyTavern users; on a multi-user server only an admin can use it. Moving it to the new storage layout gives it an owner.'
            : 'This older collection is not recorded as yours; on a multi-user server only its owner or an admin can use it. Moving it to the new storage layout gives it an owner.';
    }

    /** @returns {Promise<string|null>} a refusal message, or null when allowed */
    async function qdrantOwnerDenied(req, collectionId) {
        if (typeof collectionId !== 'string' || !qdrantBackend.baseUrl) return null;
        const name = qdrantBackend._parseCollectionName(collectionId);
        if (isReservedSharedCollection(name)) return null; // shared stores are scoped otherwise
        if (req.user?.profile?.admin === true || !await isMultiUserServer(req)) return null;
        let owner;
        try { owner = await qdrantOwnerOf(name); }
        catch (error) { return `The owner of this collection could not be checked (${error.message}); try again`; }
        if (owner === userHandle(req)) return null;
        if (owner !== undefined) return 'This collection belongs to another SillyTavern user';
        // VH-004: no owner stamp. A store that does not exist yet is created by
        // this caller (remembered); an existing one needs evidence.
        let exists = true;
        try { exists = await qdrantBackend.collectionExists(name); } catch { exists = true; }
        if (!exists) return null;
        return legacyOwnerDenied(req, name, `qdrant:${name}`);
    }

    /**
     * 9c: validates a v2 request and resolves its physical store.
     * mt-v2 without a known vector size: the tenant is searched for in the
     * existing vecthare_mt_<dim> stores (saved settings or rows); none -> null
     * (an empty collection, created at the first insert), several -> 409.
     */
    async function resolveV2Request(req, backend, logicalId, storageRaw) {
        if (backend !== 'qdrant' && backend !== 'milvus') return { error: 'This backend has no v2 storage layouts' };
        if (backend === 'qdrant' && !qdrantBackend.baseUrl) return { error: 'Qdrant not initialized', status: 503 };
        if (backend === 'milvus' && !milvusBackend.isConnected) return { error: 'Milvus not initialized', status: 503 };
        const storage = parseV2Storage(backend, storageRaw);
        if (storage.error) return { error: storage.error };
        if (!isVectHareCollection(logicalId)) return { error: 'v2 requests must name a VectHare collection' };
        const tenant = tenantFor(req, logicalId);
        if (!tenant) return { error: 'This collection id or user cannot be scoped (v2 storage refused)' };
        const configKey = configKeyForTenant(tenant.tenantKey);
        let physical = v2PhysicalName(storage.layout, tenant.tenantKey, storage.dim);
        if (storage.physical && physical && storage.physical !== physical) {
            return { error: 'Storage physical name does not match this collection' };
        }
        if (!physical && storage.layout === 'mt-v2') {
            const found = await findV2Tenant(backend, tenant.tenantKey, configKey);
            if (found.length > 1) {
                return { error: `This collection is present in ${found.join(' and ')}; refusing to guess`, status: 409 };
            }
            physical = found[0] || null;
        }
        const dim = physical && MT_V2_NAME.test(physical) ? Number(MT_V2_NAME.exec(physical)[1]) : storage.dim;
        return { backend, layout: storage.layout, physical, dim: dim ?? null, tenant, configKey };
    }

    /** vecthare_mt_<dim> stores that hold saved settings or rows for this tenant. */
    async function findV2Tenant(backend, tenantKey, configKey) {
        const names = backend === 'qdrant'
            ? await qdrantBackend.listV2CollectionNames()
            : (await milvusBackend.listCollectionNames()).filter(name => MT_V2_NAME.test(name));
        const found = [];
        for (const name of names) {
            if (backend === 'qdrant') {
                const config = await qdrantBackend.readInitialCollectionConfig(name, configKey);
                if (config !== undefined || await qdrantBackend.countTenant(name, tenantKey) > 0) found.push(name);
            } else {
                const config = await milvusBackend.readConfigV2(name, configKey);
                if (config !== undefined || await milvusBackend.countV2(name, tenantKey) > 0) found.push(name);
            }
        }
        return found;
    }

    async function withVectors(items, source, model, directories, req) {
        const missing = items.filter(item => !item.vector);
        if (!missing.length) return items;
        const vectors = await getVectorsForSource(source, missing.map(item => item.text), model, directories, req);
        let i = 0;
        return items.map(item => {
            if (item.vector) return item;
            const vector = vectors[i++];
            if (!Array.isArray(vector) || !vector.length) throw new Error(`Failed to generate embedding for item. Source: ${source}, Model: ${model}`);
            return { ...item, vector };
        });
    }

    /**
     * 9c: handler with the same method signatures as getBackendHandler(), for a
     * request already scoped by resolveV2Request (req.vh2). Every call works on
     * this tenant's data only; collectionId and filters arguments are ignored.
     */
    function getV2Handler(req) {
        const v = req.vh2;
        const { tenantKey } = v.tenant;
        const page = (items, options) => {
            const offset = options.offset || 0;
            const limit = options.limit || items.length;
            return { items: items.slice(offset, offset + limit), total: items.length, offset, limit,
                hasMore: offset + limit < items.length };
        };
        const emptyStats = { chunkCount: 0, totalCharacters: 0, totalTokens: 0, storageSize: 0, embeddingDimensions: 0,
            avgChunkSize: 0, messageCount: 0, sources: {}, backend: v.backend };
        const location = physical => ({ layout: v.layout, physical, dim: MT_V2_NAME.test(physical) ? Number(MT_V2_NAME.exec(physical)[1]) : (v.dim ?? null) });
        const insertTarget = dim => {
            if (v.dim && v.dim !== dim) {
                const error = new Error(`This collection holds ${v.dim}-dim vectors; this model makes ${dim}. Nothing was changed.`);
                error.status = 409;
                throw error;
            }
            return v.physical || (v.layout === 'mt-v2' ? mtV2Name(dim) : milvusOwnName(tenantKey));
        };

        if (v.backend === 'qdrant') {
            const scope = { tenantKey };
            return {
                type: 'qdrant', v2: true,
                list: async (_c, _s, _m, _d, options = {}) => page(v.physical ? await qdrantBackend.listItems(v.physical, scope, options) : [], options),
                get: async (_c, hash) => v.physical ? await qdrantBackend.getItem(v.physical, hash, scope) : null,
                getByHashes: async hashes => {
                    if (!v.physical) return [];
                    const wanted = new Set(hashes.map(String));
                    return (await qdrantBackend.listItems(v.physical, scope, { strictRead: true }))
                        .filter(item => wanted.has(String(item.hash)))
                        .map(item => ({ hash: item.hash, text: item.text, metadata: item.metadata }));
                },
                insert: async (_c, items, source, model, directories, request) => {
                    const ready = await withVectors(items, source, model, directories, request);
                    if (!ready.length) return v.physical ? location(v.physical) : null;
                    const physical = insertTarget(ready[0].vector.length);
                    await qdrantBackend.insertV2(physical, ready, v.tenant, { embeddingSource: source, embeddingModel: model });
                    return location(physical);
                },
                updateText: async (_c, hash, newText, source, model, directories, request) => {
                    if (!v.physical) throw new Error('Chunk not found');
                    const vector = await getEmbeddingForSource(source, newText, model, directories, request);
                    const newHash = getStringHash(newText);
                    await qdrantBackend.updateV2(v.physical, hash, { text: newText, hash: newHash, vector }, v.tenant,
                        { embeddingSource: source, embeddingModel: model });
                    return { oldHash: hash, newHash, text: newText };
                },
                updateMetadata: async (_c, hash, metadata) => {
                    if (!v.physical) throw new Error('Chunk not found');
                    await qdrantBackend.updateV2(v.physical, hash, metadata, v.tenant);
                    return { hash, metadata };
                },
                delete: async (_c, hashes) => v.physical ? await qdrantBackend.deleteV2(v.physical, hashes, tenantKey) : 0,
                query: async (_c, vector, topK, threshold) => v.physical
                    ? await qdrantBackend.queryVectors(v.physical, vector, topK, threshold, scope) : [],
                hybrid: async (vector, keywords, topK, options) => v.physical
                    ? await qdrantBackend.hybridQuery(v.physical, vector, keywords, topK, options, scope) : [],
                purge: async () => { if (v.physical) await qdrantBackend.purgeV2(v.physical, tenantKey, v.configKey); },
                stats: async () => {
                    if (!v.physical) return emptyStats;
                    const stats = await qdrantBackend.getCollectionStats(v.physical, scope);
                    delete stats.totalPoints; // the shared store's size is not this collection's
                    return stats;
                },
            };
        }
        const options = { partitionKey: v.layout === 'mt-v2' };
        return {
            type: 'milvus', v2: true,
            list: async (_c, _s, _m, _d, opts = {}) => {
                if (!v.physical) return page([], opts);
                if (opts.allHashes) {
                    const hashes = await milvusBackend.listAllHashesV2(v.physical, tenantKey);
                    return { items: hashes.map(hash => ({ hash })), total: hashes.length, offset: 0,
                        limit: hashes.length, hasMore: false, complete: true };
                }
                const items = await milvusBackend.listV2(v.physical, tenantKey, opts);
                const total = await milvusBackend.countV2(v.physical, tenantKey);
                const offset = opts.offset || 0;
                return { items, total, offset, limit: opts.limit || items.length, hasMore: offset + items.length < total };
            },
            get: async (_c, hash) => v.physical
                ? (await milvusBackend.getByHashesV2(v.physical, tenantKey, [hash], { withVector: true }))[0] || null : null,
            getByHashes: async hashes => v.physical ? await milvusBackend.getByHashesV2(v.physical, tenantKey, hashes) : [],
            insert: async (_c, items, source, model, directories, request) => {
                const ready = await withVectors(items, source, model, directories, request);
                if (!ready.length) return v.physical ? location(v.physical) : null;
                const physical = insertTarget(ready[0].vector.length);
                await milvusBackend.insertV2(physical, ready, v.tenant, { embeddingSource: source, embeddingModel: model }, options);
                return location(physical);
            },
            updateText: async (_c, hash, newText, source, model, directories, request) => {
                if (!v.physical) throw new Error('Chunk not found');
                const vector = await getEmbeddingForSource(source, newText, model, directories, request);
                const newHash = getStringHash(newText);
                await milvusBackend.updateV2(v.physical, hash, { text: newText, hash: newHash, vector }, v.tenant,
                    { embeddingSource: source, embeddingModel: model }, options);
                return { oldHash: hash, newHash, text: newText };
            },
            updateMetadata: async (_c, hash, metadata) => {
                if (!v.physical) throw new Error('Chunk not found');
                await milvusBackend.updateV2(v.physical, hash, metadata, v.tenant, {}, options);
                return { hash, metadata };
            },
            delete: async (_c, hashes) => v.physical ? await milvusBackend.deleteV2(v.physical, hashes, tenantKey) : 0,
            query: async (_c, vector, topK, threshold) => v.physical
                ? (await milvusBackend.queryV2(v.physical, tenantKey, vector, topK)).filter(r => r.score >= threshold) : [],
            hybrid: null,
            purge: async () => {
                if (v.physical) await milvusBackend.purgeV2(v.physical, tenantKey, { drop: v.layout === 'per-collection', configKey: v.configKey });
            },
            stats: async () => v.physical
                ? { chunkCount: await milvusBackend.countV2(v.physical, tenantKey), backend: 'milvus' } : { chunkCount: 0, backend: 'milvus' },
        };
    }

    function getBackendHandler(backend) {
        switch (backend) {
            case 'vectra':
            case 'standard':
                return {
                    type: 'vectra',

                    list: async (collectionId, source, model, directories, options = {}) => {
                        const store = await getIndex(directories, collectionId, source, model);
                        const items = await store.listItems();

                        // Apply pagination
                        const offset = options.offset || 0;
                        const limit = options.limit || items.length;
                        const paginatedItems = items.slice(offset, offset + limit);

                        return {
                            items: paginatedItems.map(item => ({
                                hash: item.metadata.hash,
                                text: item.metadata.text,
                                index: item.metadata.index,
                                vector: options.includeVectors ? item.vector : undefined,
                                metadata: item.metadata
                            })),
                            total: items.length,
                            offset,
                            limit,
                            hasMore: offset + limit < items.length
                        };
                    },

                    get: async (collectionId, hash, source, model, directories) => {
                        const store = await getIndex(directories, collectionId, source, model);
                        const items = await store.listItems();
                        const item = items.find(i => i.metadata.hash == hash);
                        if (!item) return null;
                        return {
                            hash: item.metadata.hash,
                            text: item.metadata.text,
                            index: item.metadata.index,
                            vector: item.vector,
                            metadata: item.metadata
                        };
                    },

                    insert: async (collectionId, items, source, model, directories, req) => {
                        const store = await getIndex(directories, collectionId, source, model);

                        // Generate embeddings if not provided
                        let itemsWithVectors = [...items];
                        const itemsNeedingVectors = itemsWithVectors.filter(i => !i.vector);

                        if (itemsNeedingVectors.length > 0) {
                            const texts = itemsNeedingVectors.map(i => i.text);
                            const vectors = await getVectorsForSource(source, texts, model, directories, req);

                            let vIndex = 0;
                            itemsWithVectors = itemsWithVectors.map(item => {
                                if (!item.vector) {
                                    return { ...item, vector: vectors[vIndex++] };
                                }
                                return item;
                            });
                        }

                        // VH-098: inserting a hash that is already stored replaces it
                        // (vectra's upsertItem without an id always appended, so every
                        // retried insert added another copy of the same chunk).
                        const incoming = new Set(itemsWithVectors.map(item => String(item.hash)));
                        const replaced = (await store.listItems())
                            .filter(existing => incoming.has(String(existing.metadata?.hash)))
                            .map(existing => existing.id);
                        await store.beginUpdate();
                        for (const id of replaced) await store.deleteItem(id);
                        for (const item of itemsWithVectors) {
                            await store.upsertItem({
                                vector: item.vector,
                                metadata: {
                                    hash: item.hash,
                                    text: item.text,
                                    index: item.index,
                                    ...item.metadata
                                }
                            });
                        }
                        await store.endUpdate();
                    },

                    updateText: async (collectionId, hash, newText, source, model, directories, req) => {
                        const store = await getIndex(directories, collectionId, source, model);
                        const items = await store.listItems();
                        const item = items.find(i => i.metadata.hash == hash);
                        if (!item) throw new Error('Chunk not found');

                        // Generate new embedding for the new text
                        const newVector = await getEmbeddingForSource(source, newText, model, directories, req);
                        const newHash = getStringHash(newText);

                        // Write the replacement first; the original is removed (by id)
                        // only after it is stored, so a failed write loses nothing.
                        await store.beginUpdate();
                        await store.upsertItem({
                            vector: newVector,
                            metadata: {
                                ...item.metadata,
                                hash: newHash,
                                text: newText
                            }
                        });
                        await store.endUpdate();
                        await store.deleteItem(item.id);

                        return { oldHash: hash, newHash, text: newText };
                    },

                    updateMetadata: async (collectionId, hash, metadata, source, model, directories) => {
                        const store = await getIndex(directories, collectionId, source, model);
                        const items = await store.listItems();
                        const item = items.find(i => i.metadata.hash == hash);
                        if (!item) throw new Error('Chunk not found');

                        // Re-insert with same vector but updated metadata, then remove
                        // the original (by id) only after the replacement is stored.
                        await store.beginUpdate();
                        await store.upsertItem({
                            vector: item.vector,
                            metadata: {
                                ...item.metadata,
                                ...metadata,
                                hash: item.metadata.hash, // Preserve hash
                                text: item.metadata.text   // Preserve text
                            }
                        });
                        await store.endUpdate();
                        await store.deleteItem(item.id);

                        return { hash, metadata };
                    },

                    delete: async (collectionId, hashes, source, model, directories) => {
                        const store = await getIndex(directories, collectionId, source, model);
                        const items = await store.listItems();

                        let deleted = 0;
                        for (const hash of hashes) {
                            const item = items.find(i => i.metadata.hash == hash);
                            if (item) {
                                await store.deleteItem(item.id);
                                deleted++;
                            }
                        }
                        return deleted;
                    },

                    query: async (collectionId, queryVector, topK, threshold, source, model, directories, options = {}) => {
                        const store = await getIndex(directories, collectionId, source, model);
                        const results = await store.queryItems(queryVector, topK);
                        return results
                            .filter(r => r.score >= threshold)
                            .map(r => ({
                                hash: r.item.metadata.hash,
                                score: r.score,
                                text: r.item.metadata.text,
                                vector: options.includeVectors ? r.item.vector : undefined,
                                metadata: r.item.metadata
                            }));
                    },

                    purge: async (collectionId, source, model, directories) => {
                        // VH-094: SillyTavern's folder (moving older plugin data there first).
                        // The saved provenance records the model VectHare asked for.
                        const requestedModel = model ? String(model) : '';
                        model = await vectraScopedModel(directories, collectionId, source, model);
                        // Do not create an index to purge it. Vectra deleteIndex()
                        // deletes its whole folder; with no model subfolder that
                        // folder may also contain *other* models' indexes.
                        const safeModel = sanitize(model);
                        if (!source || source === '.' || source === '..' ||
                            sanitize(source) !== source ||
                            (model && (!safeModel || safeModel === '.' || safeModel === '..'))) {
                            throw new Error('Unsafe Vectra source/model for scoped purge');
                        }
                        let folder = directories.vectors;
                        for (const part of [source, collectionId, ...(model ? [safeModel] : [])]) {
                            folder = path.join(folder, part);
                            const entry = await fs.lstat(folder);
                            if (!entry.isDirectory() || entry.isSymbolicLink()) {
                                throw new Error('Unsafe Vectra index directory for scoped purge');
                            }
                        }
                        let indexFile = null;
                        try {
                            indexFile = await fs.lstat(path.join(folder, 'index.json'));
                        } catch (error) {
                            // A whole collection may hold only model subfolders;
                            // those are verified below before anything is removed.
                            if (error.code !== 'ENOENT' || model) throw error;
                        }
                        if (indexFile && (!indexFile.isFile() || indexFile.isSymbolicLink())) {
                            throw new Error('Vectra index cannot be verified for scoped purge');
                        }
                        if (model && safeModel !== model) {
                            // Distinct model IDs can sanitize to the same folder.
                            // Prefer the saved VectHare provenance to identify the
                            // requested model before deleting that folder.
                            let config = null;
                            try {
                                config = JSON.parse(await fs.readFile(
                                    path.join(folder, 'vecthare-initial-config.json'), 'utf8'));
                            } catch (error) {
                                if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error;
                            }
                            if (config) {
                                if (config.collectionId !== collectionId ||
                                    config.snapshot?.embedding?.provider !== source ||
                                    ![model, requestedModel].includes(config.snapshot?.embedding?.model ?? '')) {
                                    throw new Error('Sanitized Vectra model destination is not verified');
                                }
                            } else {
                                // A legacy collection has no provenance file. The
                                // ambiguity this check guards against only exists
                                // when the collection has more than one model
                                // destination; with exactly one there is no other
                                // model index that could be confused with it, and
                                // the path already scopes the deletion to this
                                // collection and source. Otherwise, refuse.
                                const siblings = await fs.readdir(path.dirname(folder),
                                    { withFileTypes: true });
                                const models = siblings.filter(entry => entry.isDirectory() &&
                                    !entry.isSymbolicLink());
                                if (models.length !== 1 || models[0].name !== safeModel) {
                                    throw new Error('Sanitized Vectra model destination cannot be ' +
                                        'identified: no VectHare provenance file names the model this ' +
                                        'folder holds, and this collection has more than one model ' +
                                        'destination; purge blocked');
                                }
                            }
                        }
                        if (!model) {
                            // This is the parent of all model subfolders. A child
                            // is removed with it only when it is provably an EMPTY
                            // plain index (index.json with no items, optionally
                            // VectHare's config file, nothing else). Any child that
                            // holds chunks, or anything unrecognised, blocks the
                            // purge and is named, so no vectors are ever swept up.
                            const entries = await fs.readdir(folder, { withFileTypes: true });
                            const children = entries.filter(entry => entry.isDirectory() || entry.isSymbolicLink());
                            const blocking = [];
                            for (const child of children) {
                                const count = child.isSymbolicLink() ? null
                                    : await countPlainVectraIndexItems(path.join(folder, child.name));
                                if (count !== 0) {
                                    blocking.push(count === null ? `${child.name}: not a plain index`
                                        : `${child.name}: ${count} chunk${count === 1 ? '' : 's'}`);
                                }
                            }
                            if (!indexFile && (!children.length || children.length !== entries.length)) {
                                // No index of its own: only a folder of model
                                // indexes and nothing else may be removed.
                                throw new Error('Vectra index cannot be verified for scoped purge');
                            }
                            if (blocking.length) {
                                throw new Error('This collection also holds another embedding model\'s index ' +
                                    `(${blocking.join('; ')}); scoped purge blocked`);
                            }
                            for (const child of children) {
                                await fs.rm(path.join(folder, child.name), { recursive: true });
                            }
                            if (!indexFile) {
                                // rmdir only succeeds on an empty folder.
                                await fs.rmdir(folder);
                                return;
                            }
                        }
                        await new vectra.LocalIndex(folder).deleteIndex();
                    },

                    stats: async (collectionId, source, model, directories) => {
                        const store = await getIndex(directories, collectionId, source, model);
                        const items = await store.listItems();

                        let totalCharacters = 0;
                        let totalTokens = 0;
                        const sources = {};
                        const messageHashes = new Set();
                        let embeddingDimensions = 0;

                        for (const item of items) {
                            const text = item.metadata.text || '';
                            totalCharacters += text.length;
                            totalTokens += Math.ceil(text.length / 4); // Rough estimate

                            const src = item.metadata.source || 'unknown';
                            sources[src] = (sources[src] || 0) + 1;

                            if (item.metadata.originalMessageHash) {
                                messageHashes.add(item.metadata.originalMessageHash);
                            }

                            if (item.vector && item.vector.length > 0) {
                                embeddingDimensions = item.vector.length;
                            }
                        }

                        // Get file size (VH-094: in SillyTavern's folder)
                        model = await vectraScopedModel(directories, collectionId, source, model);
                        const indexPath = model
                            ? path.join(directories.vectors, sanitize(source), sanitize(collectionId), sanitize(model), 'index.json')
                            : path.join(directories.vectors, sanitize(source), sanitize(collectionId), 'index.json');

                        let storageSize = 0;
                        try {
                            const stat = await fs.stat(indexPath);
                            storageSize = stat.size;
                        } catch (e) {
                            // File may not exist yet, which is fine
                            console.debug(`[${pluginName}] Could not stat index file (may not exist): ${indexPath}`);
                        }

                        return {
                            chunkCount: items.length,
                            totalCharacters,
                            totalTokens,
                            storageSize,
                            embeddingDimensions,
                            avgChunkSize: items.length > 0 ? Math.round(totalCharacters / items.length) : 0,
                            messageCount: messageHashes.size,
                            sources,
                            backend: 'vectra',
                            model: model || '(default)'
                        };
                    }
                };

            case 'lancedb':
                return {
                    type: 'lancedb',

                    list: async (collectionId, source, model, directories, options = {}) => {
                        const lance = await lanceFor(directories);
                        const items = await lance.listItems(collectionId, source, options);
                        // Return same format as Vectra handler for consistency
                        const offset = options.offset || 0;
                        const limit = options.limit || items.length;
                        const paginatedItems = items.slice(offset, offset + limit);
                        return {
                            items: paginatedItems,
                            total: items.length,
                            offset,
                            limit,
                            hasMore: offset + limit < items.length
                        };
                    },

                    get: async (collectionId, hash, source, model, directories) => {
                        return await (await lanceFor(directories)).getItem(collectionId, hash, source);
                    },

                    insert: async (collectionId, items, source, model, directories, req) => {
                        const lance = await lanceFor(directories);
                        // Generate embeddings if not provided
                        let itemsWithVectors = [...items];
                        const itemsNeedingVectors = itemsWithVectors.filter(i => !i.vector);

                        if (itemsNeedingVectors.length > 0) {
                            console.log(`[LanceDB] Generating embeddings for ${itemsNeedingVectors.length} items`);
                            const texts = itemsNeedingVectors.map(i => i.text);
                            const vectors = await getVectorsForSource(source, texts, model, directories, req);

                            let vIndex = 0;
                            itemsWithVectors = itemsWithVectors.map(item => {
                                if (!item.vector) {
                                    const vector = vectors[vIndex++];
                                    if (!vector || !Array.isArray(vector) || vector.length === 0) {
                                        console.error(`[LanceDB] Failed to generate valid vector for item hash=${item.hash}, source=${source}, model=${model}`);
                                        throw new Error(`Failed to generate embedding for item. Source: ${source}, Model: ${model}`);
                                    }
                                    return { ...item, vector };
                                }
                                return item;
                            });
                        }

                        await lance.insertVectors(collectionId, itemsWithVectors, source);
                    },

                    updateText: async (collectionId, hash, newText, source, model, directories, req) => {
                        const lance = await lanceFor(directories);
                        // Get new embedding
                        const newVector = await getEmbeddingForSource(source, newText, model, directories, req);
                        const newHash = getStringHash(newText);
                        await lance.updateItem(collectionId, hash, { text: newText, hash: newHash, vector: newVector }, source);
                        return { oldHash: hash, newHash, text: newText };
                    },

                    updateMetadata: async (collectionId, hash, metadata, source, model, directories) => {
                        await (await lanceFor(directories)).updateItemMetadata(collectionId, hash, metadata, source);
                        return { hash, metadata };
                    },

                    delete: async (collectionId, hashes, source, model, directories) => {
                        const safe = toSafeHashList(hashes);
                        await (await lanceFor(directories)).deleteVectors(collectionId, safe, source);
                        return safe.length;
                    },

                    query: async (collectionId, queryVector, topK, threshold, source, model, directories, options = {}) => {
                        const lance = await lanceFor(directories);
                        return await lance.queryVectors(collectionId, queryVector, topK, threshold, source);
                    },

                    purge: async (collectionId, source, model, directories) => {
                        await (await lanceFor(directories)).purgeCollection(collectionId, source);
                    },

                    stats: async (collectionId, source, model, directories) => {
                        return await (await lanceFor(directories)).getCollectionStats(collectionId, source);
                    }
                };

            case 'qdrant':
                return {
                    type: 'qdrant',

                    list: async (collectionId, source, model, directories, options = {}) => {
                        const items = await qdrantBackend.listItems(collectionId, options.filters || {}, options);
                        // Return same format as Vectra handler for consistency
                        const offset = options.offset || 0;
                        const limit = options.limit || items.length;
                        const paginatedItems = items.slice(offset, offset + limit);
                        return {
                            items: paginatedItems,
                            total: items.length,
                            offset,
                            limit,
                            hasMore: offset + limit < items.length
                        };
                    },

                    get: async (collectionId, hash, source, model, directories, filters = {}) => {
                        return await qdrantBackend.getItem(collectionId, hash, filters);
                    },

                    insert: async (collectionId, items, source, model, directories, req, filters = {}) => {
                        // Generate embeddings if not provided
                        let itemsWithVectors = [...items];
                        const itemsNeedingVectors = itemsWithVectors.filter(i => !i.vector);

                        if (itemsNeedingVectors.length > 0) {
                            console.log(`[Qdrant] Generating embeddings for ${itemsNeedingVectors.length} items`);
                            const texts = itemsNeedingVectors.map(i => i.text);
                            const vectors = await getVectorsForSource(source, texts, model, directories, req);

                            let vIndex = 0;
                            itemsWithVectors = itemsWithVectors.map(item => {
                                if (!item.vector) {
                                    const vector = vectors[vIndex++];
                                    if (!vector || !Array.isArray(vector) || vector.length === 0) {
                                        console.error(`[Qdrant] Failed to generate valid vector for item hash=${item.hash}, source=${source}, model=${model}`);
                                        throw new Error(`Failed to generate embedding for item. Source: ${source}, Model: ${model}`);
                                    }
                                    return { ...item, vector };
                                }
                                return item;
                            });
                        }

                        // Pass source and model for embedding tracking
                        await qdrantBackend.insertVectors(collectionId, itemsWithVectors, {
                            ...filters,
                            embeddingSource: source,
                            embeddingModel: model,
                        });
                    },

                    updateText: async (collectionId, hash, newText, source, model, directories, req, filters = {}) => {
                        const newVector = await getEmbeddingForSource(source, newText, model, directories, req);
                        const newHash = getStringHash(newText);
                        await qdrantBackend.updateItem(collectionId, hash, { text: newText, hash: newHash, vector: newVector }, filters);
                        return { oldHash: hash, newHash, text: newText };
                    },

                    updateMetadata: async (collectionId, hash, metadata, source, model, directories, filters = {}) => {
                        await qdrantBackend.updateItemMetadata(collectionId, hash, metadata, filters);
                        return { hash, metadata };
                    },

                    delete: async (collectionId, hashes, source, model, directories, filters = {}) => {
                        await qdrantBackend.deleteVectors(collectionId, hashes, filters);
                        return hashes.length;
                    },

                    query: async (collectionId, queryVector, topK, threshold, source, model, directories, options = {}) => {
                        const results = await qdrantBackend.queryVectors(collectionId, queryVector, topK, threshold, options.filters || {});
                        return results;
                    },

                    purge: async (collectionId, source, model, directories, filters = {}) => {
                        await qdrantBackend.purgeAll(collectionId, filters);
                    },

                    stats: async (collectionId, source, model, directories, filters = {}) => {
                        return await qdrantBackend.getCollectionStats(collectionId, filters);
                    }
                };

            case 'milvus':
                return {
                    type: 'milvus',

                    list: async (collectionId, source, model, directories, options = {}) => {
                        if (options.allHashes) {
                            const hashes = await milvusBackend.listAllHashes(options.filters || {});
                            return { items: hashes.map(hash => ({ hash })), total: hashes.length, offset: 0,
                                limit: hashes.length, hasMore: false, complete: true };
                        }
                        const items = await milvusBackend.listItems(collectionId, options.filters || {}, options);
                        const offset = options.offset || 0;
                        const limit = options.limit || items.length;
                        // Report the collection's real size, not the page length, so
                        // callers can tell a complete listing from a truncated one.
                        let total = null;
                        try { total = await milvusBackend.countItems(options.filters || {}); }
                        catch (error) { if (options.strictRead) throw error; }
                        return {
                            items: items,
                            total: total ?? offset + items.length,
                            offset,
                            limit,
                            hasMore: total !== null ? offset + items.length < total : items.length >= limit
                        };
                    },

                    get: async (collectionId, hash, source, model, directories, filters = {}) => {
                        return await milvusBackend.getItem(collectionId, hash, filters);
                    },

                    insert: async (collectionId, items, source, model, directories, req, filters = {}) => {
                        // Generate embeddings if not provided
                        let itemsWithVectors = [...items];
                        const itemsNeedingVectors = itemsWithVectors.filter(i => !i.vector);

                        if (itemsNeedingVectors.length > 0) {
                            console.log(`[Milvus] Generating embeddings for ${itemsNeedingVectors.length} items`);
                            const texts = itemsNeedingVectors.map(i => i.text);
                            const vectors = await getVectorsForSource(source, texts, model, directories, req);

                            let vIndex = 0;
                            itemsWithVectors = itemsWithVectors.map(item => {
                                if (!item.vector) {
                                    return { ...item, vector: vectors[vIndex++] };
                                }
                                return item;
                            });
                        }

                        await milvusBackend.insertVectors(collectionId, itemsWithVectors, {
                            ...filters,
                            embeddingSource: source,
                            embeddingModel: model,
                        });
                    },

                    updateText: async (collectionId, hash, newText, source, model, directories, req, filters = {}) => {
                        const newVector = await getEmbeddingForSource(source, newText, model, directories, req);
                        const newHash = getStringHash(newText);
                        await milvusBackend.updateItem(collectionId, hash, { text: newText, hash: newHash, vector: newVector }, filters);
                        return { oldHash: hash, newHash, text: newText };
                    },

                    updateMetadata: async (collectionId, hash, metadata, source, model, directories, filters = {}) => {
                        await milvusBackend.updateItem(collectionId, hash, metadata, filters);
                        return { hash, metadata };
                    },

                    delete: async (collectionId, hashes, source, model, directories, filters = {}) => {
                        await milvusBackend.deleteVectors(collectionId, hashes, filters);
                        return hashes.length;
                    },

                    query: async (collectionId, queryVector, topK, threshold, source, model, directories, options = {}) => {
                        const results = await milvusBackend.queryCollection(collectionId, queryVector, topK, options.filters || {});
                        return results.filter(r => r.score >= threshold);
                    },

                    purge: async (collectionId, source, model, directories, filters = {}) => {
                        // VH-011: reports how many shared rows had to stay.
                        return await milvusBackend.purgeCollection(collectionId, filters);
                    },

                    stats: async (collectionId, source, model, directories, filters = {}) => {
                        return await milvusBackend.getCollectionStats(collectionId, filters);
                    }
                };

            default:
                throw new Error(`Unknown backend: ${backend}`);
        }
    }

    // ========================================================================
    // UTILITY ENDPOINTS
    // ========================================================================

    /**
     * GET /api/plugins/similharity/health
     * Overall plugin health check
     */
    router.get('/health', (req, res) => {
        res.json({
            status: 'ok',
            plugin: pluginName,
            version: pluginVersion,
            protocol: PLUGIN_PROTOCOL,
            backends: ['vectra', 'lancedb', 'qdrant', 'milvus'],
            capabilities: { ownedStateTokens: 1, ledgerTokens: 1, leases: 1, storageReport: 1, scopedTenancy: V2_CAPABILITY },
        });
    });

    /**
     * POST /api/plugins/similharity/collections/paths
     * Where a collection and VectHare's metadata document actually live, for
     * display. READ-ONLY: it resolves, checks and reports; it creates nothing
     * and returns null for anything it cannot confirm. Remote backends have no
     * local path and say so.
     */
    router.post('/collections/paths', async (req, res) => {
        try {
            const { backend = 'vectra', collectionId, source, model } = req.body;
            if (typeof collectionId !== 'string' || !collectionId || collectionId.length > 255 ||
                /[\\/]|\.\.|[\u0000-\u001f]/.test(collectionId)) {
                return res.status(400).json({ error: 'Invalid collectionId' });
            }
            const vectorsPath = req.user.directories.vectors;
            const exists = async target => {
                try { await fs.stat(target); return true; } catch { return false; }
            };

            // VectHare's own metadata document, when it has been written.
            const metadataFile = path.join(vectorsPath, 'vecthare-owned-state', 'collections.json');
            const metadata = await exists(metadataFile) ? metadataFile : null;

            if (backend === 'qdrant' || backend === 'milvus') {
                return res.json({ success: true, remote: true, collection: null, metadata });
            }

            const safeSource = sanitize(String(source || ''));
            if (!safeSource || safeSource !== String(source || '')) {
                return res.json({ success: true, remote: false, collection: null, metadata });
            }
            let candidate;
            if (backend === 'lancedb') {
                candidate = path.join(vectorsPath, 'lancedb', safeSource, `${collectionId}.lance`);
            } else {
                // VH-094: SillyTavern's model folder for this source.
                const scoped = sanitize(String(collectionId)) === String(collectionId)
                    ? await vectraScopedModel(req.user.directories, collectionId, safeSource, String(model || '')) : String(model || '');
                const safeModel = sanitize(scoped);
                candidate = path.join(vectorsPath, safeSource, collectionId,
                    ...(safeModel ? [safeModel] : []));
            }
            return res.json({
                success: true, remote: false, metadata,
                collection: await exists(candidate) ? candidate : null,
            });
        } catch (error) {
            console.error(`[${pluginName}] collections/paths error:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/open-folder
     * Opens collection folder in file explorer
     */
    router.post('/open-folder', async (req, res) => {
        try {
            const { collectionId, backend, source } = req.body;

            if (!collectionId || !backend) {
                return res.status(400).json({ error: 'collectionId and backend are required' });
            }

            const vectorsPath = req.user.directories.vectors;
            let folderPath;

            if (backend === 'lancedb') {
                // LanceDB structure: lancedb/{source}/{collectionId}.lance
                const effectiveSource = source || 'bananabread';
                folderPath = path.join(vectorsPath, 'lancedb', effectiveSource, `${collectionId}.lance`);
            } else if (backend === 'qdrant' || backend === 'milvus') {
                // Remote backends have no folder on this machine. Milvus used
                // to fall through and build a Vectra-shaped path that could
                // never exist, reporting a confusing "folder not found".
                return res.status(400).json({
                    error: `${backend === 'qdrant' ? 'Qdrant' : 'Milvus'} collections are stored on the ` +
                        'server, not in a folder on this machine',
                });
            } else {
                const effectiveSource = source || 'transformers';
                folderPath = path.join(vectorsPath, effectiveSource, collectionId);
            }

            folderPath = path.resolve(folderPath);
            // Open Folder is allowed to READ only inside the user's vectors
            // directory. A path or symlink must not escape that root.
            let realFolder;
            try {
                realFolder = await fs.realpath(folderPath);
                if (!(await fs.stat(realFolder)).isDirectory()) throw new Error('Not a directory');
            } catch {
                return res.status(404).json({ error: `Folder not found: ${folderPath}` });
            }
            const realRoot = await fs.realpath(vectorsPath);
            if (!realFolder.startsWith(realRoot + path.sep)) {
                return res.status(403).json({ error: 'Folder is outside the vectors directory' });
            }
            // VH-060 (user decision a): a file manager opens on the machine that
            // runs SillyTavern, so only for a request made on that machine, or by
            // an admin. Anyone else (e.g. a remote user of a shared server) gets
            // the path to look at instead. A request that came through a proxy
            // (forwarding headers) is not treated as local even from 127.0.0.1.
            const remote = String(req.socket?.remoteAddress || '');
            const viaProxy = Boolean(req.headers?.['x-forwarded-for'] || req.headers?.['forwarded'] || req.headers?.['x-real-ip']);
            const local = !viaProxy && (remote === '127.0.0.1' || remote === '::1' || remote === '::ffff:127.0.0.1');
            if (!local && req.user?.profile?.admin !== true) {
                return res.json({ success: true, opened: false, path: realFolder,
                    reason: 'The folder is on the SillyTavern server, not on this device' });
            }
            const platform = process.platform;
            const command = platform === 'win32' ? 'explorer.exe' : platform === 'darwin' ? 'open' : 'xdg-open';
            const child = spawn(command, [realFolder], { shell: false, detached: true, stdio: 'ignore' });
            child.on('error', error => console.warn(`[${pluginName}] Open folder failed:`, error.message));
            child.unref();
            res.json({ success: true, opened: true, path: realFolder });

        } catch (error) {
            console.error(`[${pluginName}] open-folder error:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    // Stage 13.0: compact, VectHare-owned source-identity ledger. Never write
    // into ST chat files or modify a vanilla-generated index.
    function identityLedgerPath(req, collectionId, binding) {
        if (!isVectHareCollection(collectionId) || !binding ||
            !['standard', 'vectra', 'lancedb', 'qdrant', 'milvus'].includes(binding.backend) ||
            typeof binding.source !== 'string' || !binding.source ||
            typeof binding.model !== 'string' || !binding.storage ||
            typeof binding.storage !== 'object') {
            throw new Error('Invalid VectHare identity ledger destination');
        }
        const root = path.join(req.user.directories.vectors, 'vecthare-owned-identity-ledgers');
        const key = createHash('sha256')
            .update(JSON.stringify([collectionId, binding])).digest('hex');
        return { root, filename: path.join(root, `${key}.json`) };
    }

    async function ensureIdentityLedgerDirectory(root) {
        await fs.mkdir(root, { recursive: true });
        const stat = await fs.lstat(root);
        const parent = await fs.realpath(path.dirname(root));
        if (!stat.isDirectory() || stat.isSymbolicLink() ||
            await fs.realpath(root) !== path.join(parent, 'vecthare-owned-identity-ledgers')) {
            throw new Error('Unsafe VectHare identity ledger storage directory');
        }
    }

    function validateIdentityLedger(ledger, collectionId, binding) {
        if (!ledger || ledger.version !== 1 || ledger.collectionId !== collectionId ||
            JSON.stringify(ledger.binding) !== JSON.stringify(binding) ||
            !['clean', 'dirty'].includes(ledger.state) ||
            (ledger.state === 'clean' && ledger.pending != null) ||
            !Array.isArray(ledger.messages) || !Array.isArray(ledger.groups) ||
            (ledger.state === 'dirty' && (!ledger.pending ||
                !Array.isArray(ledger.pending.insertHashes) || !Array.isArray(ledger.pending.deleteHashes)))) {
            throw new Error('Invalid or mismatched identity ledger');
        }
        const uids = new Set();
        for (const msg of ledger.messages) {
            if (typeof msg.uid !== 'string' || !/^m_[0-9a-f]{32}$/.test(msg.uid) ||
                uids.has(msg.uid) || typeof msg.fingerprint !== 'string' ||
                !/^[0-9a-f]{64}$/.test(msg.fingerprint) ||
                !Number.isSafeInteger(msg.position) || msg.position < 0 ||
                typeof msg.role !== 'string' || typeof msg.name !== 'string' ||
                (msg.retiredConfirmed !== undefined && typeof msg.retiredConfirmed !== 'boolean')) {
                throw new Error('Invalid identity ledger message');
            }
            uids.add(msg.uid);
        }
        for (const group of ledger.groups) {
            if (!Array.isArray(group.memberUids) || !group.memberUids.length ||
                !group.memberUids.every(uid => uids.has(uid)) ||
                (group.provenanceUids !== undefined &&
                    (!Array.isArray(group.provenanceUids) ||
                        !group.provenanceUids.length ||
                        new Set(group.provenanceUids).size !== group.provenanceUids.length ||
                        !group.provenanceUids.every(uid =>
                            typeof uid === 'string' && /^m_[0-9a-f]{32}$/.test(uid)))) ||
                !['string', 'number'].includes(typeof group.hash) ||
                !/^[0-9a-f]{64}$/.test(group.hashInputFingerprint) ||
                typeof group.indexed !== 'boolean') {
                throw new Error('Invalid identity ledger group');
            }
        }
    }

    // Step 8C: short server-side leases so two tabs/devices do not run the
    // same work on one collection at once. In memory only (a restart clears
    // them, which is harmless); a lease expires LEASE_TTL_MS after its last
    // renewal, so a closed or crashed tab blocks nobody for long. `*` is an
    // exclusive per-user lease (Purge All): it can only be taken while no
    // other lease is held, and blocks every other lease while held.
    const LEASE_TTL_MS = 180 * 1000;
    const leases = new Map(); // `${vectorsDir}\u0000${collectionId}` -> { owner, token, expires, label }
    const leaseKey = (req, collectionId) => `${req.user.directories.vectors}\u0000${collectionId}`;
    function liveLeases(req) {
        const prefix = `${req.user.directories.vectors}\u0000`;
        const now = Date.now();
        const out = [];
        for (const [key, lease] of leases) {
            if (lease.expires <= now) { leases.delete(key); continue; }
            if (key.startsWith(prefix)) out.push([key.slice(prefix.length), lease]);
        }
        return out;
    }
    const validLeaseTarget = id => id === '*' || isVectHareCollection(id);

    router.post('/locks/acquire', (req, res) => {
        const { collectionId, owner, label } = req.body || {};
        if (!validLeaseTarget(collectionId) || typeof owner !== 'string' || !owner || owner.length > 128) {
            return res.status(400).json({ error: 'Invalid lease request' });
        }
        const live = liveLeases(req);
        const exclusive = live.find(([id]) => id === '*');
        const mine = lease => lease.owner === owner;
        if (exclusive && !mine(exclusive[1])) {
            return res.status(409).json({ held: true, label: exclusive[1].label || 'Purge All' });
        }
        if (collectionId === '*' && live.some(([id, lease]) => id !== '*' && !mine(lease))) {
            return res.status(409).json({ held: true, label: 'other work in progress' });
        }
        const key = leaseKey(req, collectionId);
        const existing = leases.get(key);
        if (existing && existing.expires > Date.now() && !mine(existing)) {
            return res.status(409).json({ held: true, label: existing.label || '' });
        }
        const lease = existing && mine(existing) && existing.expires > Date.now()
            ? existing : { owner, token: randomUUID(), label: typeof label === 'string' ? label.slice(0, 80) : '' };
        lease.expires = Date.now() + LEASE_TTL_MS;
        leases.set(key, lease);
        return res.json({ success: true, token: lease.token, ttlMs: LEASE_TTL_MS });
    });

    router.post('/locks/renew', (req, res) => {
        const { collectionId, token } = req.body || {};
        liveLeases(req); // drop expired
        const lease = leases.get(leaseKey(req, collectionId));
        if (!lease || lease.token !== token) return res.status(409).json({ lost: true });
        lease.expires = Date.now() + LEASE_TTL_MS;
        return res.json({ success: true, ttlMs: LEASE_TTL_MS });
    });

    router.post('/locks/release', (req, res) => {
        const { collectionId, token } = req.body || {};
        const key = leaseKey(req, collectionId);
        const lease = leases.get(key);
        if (lease && lease.token === token) leases.delete(key);
        return res.json({ success: true });
    });

    // Step 8B: ledger files carry a server-owned revision; reads return a
    // token (`missing`, `r<revision>`, `x<hash>` for an unreadable file) and a
    // save naming the token it was based on is refused with 409 if the file
    // changed meanwhile - so a second tab cannot overwrite a ledger another
    // tab staged. A save without a token (older builds) behaves as before.
    const ledgerLocks = new Map();
    async function withLedgerLock(key, task) {
        const previous = ledgerLocks.get(key) || Promise.resolve();
        let release;
        const current = new Promise(resolve => { release = resolve; });
        const chained = previous.then(() => current);
        ledgerLocks.set(key, chained);
        await previous.catch(() => {});
        try { return await task(); }
        finally {
            release();
            if (ledgerLocks.get(key) === chained) ledgerLocks.delete(key);
        }
    }

    async function readLedgerFile(filename, collectionId, binding) {
        let raw;
        try { raw = await fs.readFile(filename, 'utf8'); }
        catch (error) {
            if (error.code === 'ENOENT') return { status: 'missing', token: 'missing' };
            throw error;
        }
        const invalid = { status: 'invalid', token: 'x' + createHash('sha256').update(raw).digest('hex').slice(0, 16) };
        let ledger;
        try { ledger = JSON.parse(raw); } catch { return invalid; }
        try { validateIdentityLedger(ledger, collectionId, binding); } catch { return invalid; }
        const revision = Number.isSafeInteger(ledger.revision) && ledger.revision >= 0 ? ledger.revision : 0;
        return { status: 'found', ledger, revision, token: `r${revision}` };
    }

    router.post('/identity-ledger/read', requireVectHareCollection, async (req, res) => {
        try {
            const { collectionId, binding } = req.body;
            const { filename } = identityLedgerPath(req, collectionId, binding);
            const current = await readLedgerFile(filename, collectionId, binding);
            return res.json({ success: true, status: current.status, token: current.token,
                ...(current.status === 'found' ? { ledger: current.ledger } : {}) });
        } catch (error) {
            return res.status(400).json({ error: error.message });
        }
    });

    router.post('/identity-ledger/save', requireVectHareCollection, async (req, res) => {
        try {
            const { collectionId, binding, ledger, baseToken } = req.body;
            const { root, filename } = identityLedgerPath(req, collectionId, binding);
            if (ledger !== null) validateIdentityLedger(ledger, collectionId, binding);
            return await withLedgerLock(filename, async () => {
                let current = null;
                if (baseToken !== undefined) {
                    current = await readLedgerFile(filename, collectionId, binding);
                    if (baseToken !== current.token) {
                        return res.status(409).json({ error: 'Identity ledger changed since it was read',
                            conflict: true, token: current.token });
                    }
                }
                // Null means explicitly invalidating tracking after a confirmed purge.
                if (ledger === null) {
                    // The invalidate path must meet the same anti-symlink rule as
                    // writes, even when the directory was replaced by another app.
                    try { await fs.lstat(root); await ensureIdentityLedgerDirectory(root); }
                    catch (error) {
                        if (error.code === 'ENOENT') return res.json({ success: true, token: 'missing' });
                        throw error;
                    }
                    await fs.unlink(filename).catch(error => {
                        if (error.code !== 'ENOENT') throw error;
                    });
                    return res.json({ success: true, token: 'missing' });
                }
                current ??= await readLedgerFile(filename, collectionId, binding);
                const revision = (current.status === 'found' ? current.revision : 0) + 1;
                const serialized = JSON.stringify({ ...ledger, revision });
                if (Buffer.byteLength(serialized, 'utf8') > 16 * 1024 * 1024) {
                    return res.status(413).json({ error: 'Identity ledger too large' });
                }
                await ensureIdentityLedgerDirectory(root);
                const temporary = path.join(root, `${path.basename(filename)}.${randomUUID()}.tmp`);
                try {
                    await fs.writeFile(temporary, serialized, { flag: 'wx', encoding: 'utf8' });
                    await fs.rename(temporary, filename);
                } finally {
                    await fs.unlink(temporary).catch(() => {});
                }
                return res.json({ success: true, token: `r${revision}` });
            });
        } catch (error) {
            return res.status(400).json({ error: error.message });
        }
    });

    // ========================================================================
    // VECTHARE-OWNED LOCAL STATE (collections.json)
    // ========================================================================
    // Phase 1: VectHare's own registry, collection metadata, chat bindings and
    // chunk annotations live in one authoritative per-user file instead of
    // SillyTavern's extension settings. Identity ledgers keep their own files
    // and the authoritative original vectorization configuration stays
    // backend-resident; neither is stored here.

    const OWNED_STATE_DIRECTORY = 'vecthare-owned-state';
    const OWNED_STATE_FILE = 'collections.json';
    const OWNED_STATE_MAX_BYTES = 16 * 1024 * 1024;

    function ownedStatePath(req) {
        const root = path.join(req.user.directories.vectors, OWNED_STATE_DIRECTORY);
        return { root, filename: path.join(root, OWNED_STATE_FILE) };
    }

    async function ensureOwnedStateDirectory(root) {
        await fs.mkdir(root, { recursive: true });
        const stat = await fs.lstat(root);
        const parent = await fs.realpath(path.dirname(root));
        if (!stat.isDirectory() || stat.isSymbolicLink() ||
            await fs.realpath(root) !== path.join(parent, OWNED_STATE_DIRECTORY)) {
            throw new Error('Unsafe VectHare state storage directory');
        }
    }

    const isPlainObject = value => Boolean(value) && typeof value === 'object' && !Array.isArray(value);

    function validateOwnedState(state) {
        if (!isPlainObject(state) || state.version !== 1 ||
            !Array.isArray(state.registry) || !state.registry.every(id => typeof id === 'string') ||
            !isPlainObject(state.collections) || !isPlainObject(state.chunkMetadata) ||
            !isPlainObject(state.chatCollectionPolicies) ||
            !isPlainObject(state.chatCollectionBindings) ||
            !isPlainObject(state.chatCollectionBindings.byCollection)) {
            throw new Error('Invalid VectHare state document');
        }
    }

    // Step 8A: collections.json carries a server-owned revision. Every read
    // returns a token for the file as it is (`missing`, `r<revision>`, or
    // `x<hash>` for an unreadable file); a write that names the token it was
    // based on is refused with 409 when the file has changed since, so a stale
    // tab can no longer overwrite another tab's changes. A write without a
    // token (older extension builds) behaves as before.
    const ownedStateLocks = new Map();
    async function withOwnedStateLock(key, task) {
        const previous = ownedStateLocks.get(key) || Promise.resolve();
        let release;
        const current = new Promise(resolve => { release = resolve; });
        const chained = previous.then(() => current);
        ownedStateLocks.set(key, chained);
        await previous.catch(() => {});
        try { return await task(); }
        finally {
            release();
            if (ownedStateLocks.get(key) === chained) ownedStateLocks.delete(key);
        }
    }

    async function readOwnedStateFile(filename) {
        let raw;
        try {
            raw = await fs.readFile(filename, 'utf8');
        } catch (error) {
            if (error.code === 'ENOENT') return { status: 'missing', token: 'missing' };
            throw error;
        }
        let state;
        let migratedFrom = null;
        try {
            state = JSON.parse(raw);
            // VH-063: a document from an earlier 13.3x build may lack top-level
            // keys added later. Missing keys get their empty defaults (written
            // back by the next save); wrong types are still refused.
            if (isPlainObject(state)) {
                const added = [];
                if (state.version === undefined) { state.version = 1; added.push('version'); }
                for (const [key, empty] of [['registry', []], ['collections', {}], ['chunkMetadata', {}], ['chatCollectionPolicies', {}]]) {
                    if (state[key] === undefined) { state[key] = empty; added.push(key); }
                }
                if (state.chatCollectionBindings === undefined) {
                    state.chatCollectionBindings = { version: 1, byCollection: {} }; added.push('chatCollectionBindings');
                } else if (isPlainObject(state.chatCollectionBindings) && state.chatCollectionBindings.byCollection === undefined) {
                    state.chatCollectionBindings.byCollection = {}; added.push('chatCollectionBindings.byCollection');
                }
                if (added.length) migratedFrom = added;
            }
            validateOwnedState(state);
        } catch (error) {
            return { status: 'invalid', reason: error.message,
                token: 'x' + createHash('sha256').update(raw).digest('hex').slice(0, 16) };
        }
        const revision = Number.isSafeInteger(state.revision) && state.revision >= 0 ? state.revision : 0;
        return { status: 'found', state, revision, token: `r${revision}`, ...(migratedFrom ? { migratedFrom } : {}) };
    }

    router.post('/owned-state/read', async (req, res) => {
        try {
            const { filename } = ownedStatePath(req);
            // A missing file is a fact the caller must be able to act on, and
            // unreadable content is never reported as an empty store.
            const current = await readOwnedStateFile(filename);
            return res.json({ success: true, ...current, serverNow: Date.now() }); // VH-065: the server's clock
        } catch (error) {
            console.error('[Similharity] Failed to read VectHare state:', error);
            return res.status(500).json({ error: error.message });
        }
    });

    router.post('/owned-state/token', async (req, res) => {
        try {
            const { filename } = ownedStatePath(req);
            const current = await readOwnedStateFile(filename);
            return res.json({ success: true, token: current.token, serverNow: Date.now() });
        } catch (error) {
            return res.status(500).json({ error: error.message });
        }
    });

    router.post('/owned-state/write', async (req, res) => {
        try {
            const { state, baseToken } = req.body;
            validateOwnedState(state);
            const { root, filename } = ownedStatePath(req);
            return await withOwnedStateLock(filename, async () => {
                const current = await readOwnedStateFile(filename);
                if (baseToken !== undefined && baseToken !== current.token) {
                    return res.status(409).json({ error: 'VectHare state changed since it was read',
                        conflict: true, token: current.token, status: current.status,
                        state: current.status === 'found' ? current.state : null });
                }
                const revision = (current.status === 'found' ? current.revision : 0) + 1;
                const serialized = JSON.stringify({ ...state, revision });
                if (Buffer.byteLength(serialized, 'utf8') > OWNED_STATE_MAX_BYTES) {
                    return res.status(413).json({ error: 'VectHare state document too large' });
                }
                await ensureOwnedStateDirectory(root);
                const temporary = path.join(root, `${OWNED_STATE_FILE}.${randomUUID()}.tmp`);
                try {
                    await fs.writeFile(temporary, serialized, { flag: 'wx', encoding: 'utf8' });
                    await fs.rename(temporary, filename);
                } finally {
                    await fs.unlink(temporary).catch(() => {});
                }
                // Read back before reporting success: a write the caller cannot
                // re-read is not a write it may treat as authoritative.
                const verification = await fs.readFile(filename, 'utf8');
                if (verification !== serialized) {
                    throw new Error('VectHare state verification failed after write');
                }
                return res.json({ success: true, verified: true, serverNow: Date.now(), token: `r${revision}`,
                    bytes: Buffer.byteLength(serialized, 'utf8') });
            });
        } catch (error) {
            console.error('[Similharity] Failed to write VectHare state:', error);
            return res.status(400).json({ error: error.message });
        }
    });

    // VectHare key store (Step 7 / U1). Qdrant/Milvus credentials live in one
    // per-user file beside collections.json instead of SillyTavern's
    // settings.json. There is deliberately NO route that returns a value: the
    // browser can only learn whether a name is set, replace it, or remove it.
    const SECRETS_FILE = 'secrets.json';
    const SECRET_NAMES = ['qdrant_api_key', 'milvus_token', 'milvus_password'];
    const SECRET_MAX_CHARS = 4096;
    let secretsWriteChain = Promise.resolve();

    function secretsPath(req) {
        const root = path.join(req.user.directories.vectors, OWNED_STATE_DIRECTORY);
        return { root, filename: path.join(root, SECRETS_FILE) };
    }

    /** Reads the key file. Missing = {}. Unreadable/invalid THROWS (never treated as "no keys"). */
    async function readSecretsFile(req) {
        const { filename } = secretsPath(req);
        let raw;
        try {
            raw = await fs.readFile(filename, 'utf8');
        } catch (error) {
            if (error.code === 'ENOENT') return {};
            throw error;
        }
        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch {
            throw new Error('VectHare key store is unreadable; fix or remove secrets.json');
        }
        if (!isPlainObject(parsed)) throw new Error('VectHare key store is invalid');
        const clean = {};
        for (const name of SECRET_NAMES) {
            if (typeof parsed[name] === 'string' && parsed[name]) clean[name] = parsed[name];
        }
        return clean;
    }

    async function writeSecretsFile(req, secrets) {
        const { root, filename } = secretsPath(req);
        await ensureOwnedStateDirectory(root);
        const serialized = JSON.stringify(secrets);
        const temporary = path.join(root, `${SECRETS_FILE}.${randomUUID()}.tmp`);
        try {
            await fs.writeFile(temporary, serialized, { flag: 'wx', encoding: 'utf8', mode: 0o600 });
            await fs.rename(temporary, filename);
        } finally {
            await fs.unlink(temporary).catch(() => {});
        }
        await fs.chmod(filename, 0o600).catch(() => {}); // not supported everywhere
        const verification = await fs.readFile(filename, 'utf8');
        if (verification !== serialized) throw new Error('VectHare key store verification failed after write');
    }

    const secretStatus = secrets => Object.fromEntries(SECRET_NAMES.map(name => [name, Boolean(secrets[name])]));

    router.post('/secrets/status', async (req, res) => {
        try {
            return res.json({ success: true, status: secretStatus(await readSecretsFile(req)) });
        } catch (error) {
            console.error('[Similharity] Failed to read key store status:', error.message);
            return res.status(500).json({ error: error.message });
        }
    });

    router.post('/secrets/set', async (req, res) => {
        try {
            const { name, value } = req.body || {};
            if (!SECRET_NAMES.includes(name)) return res.status(400).json({ error: 'Unknown secret name' });
            if (typeof value !== 'string') return res.status(400).json({ error: 'Secret value must be a string' });
            if (value.length > SECRET_MAX_CHARS) return res.status(413).json({ error: 'Secret value too long' });
            // Serialised so two quick saves cannot overwrite each other's name.
            const run = secretsWriteChain.then(async () => {
                const secrets = await readSecretsFile(req); // throws if unreadable: never overwrite blind
                if (value) secrets[name] = value; else delete secrets[name];
                await writeSecretsFile(req, secrets);
                return secrets;
            });
            secretsWriteChain = run.catch(() => {});
            const secrets = await run;
            return res.json({ success: true, verified: true, status: secretStatus(secrets) });
        } catch (error) {
            console.error('[Similharity] Failed to write key store:', error.message);
            return res.status(500).json({ error: error.message });
        }
    });

    /**
     * Applies stored credentials to a backend init config. A stored value wins
     * over the request body (old extension builds that still send keys keep
     * working until they migrate). Qdrant's key is only applied to a URL/cloud
     * target or when the caller already supplied a key, never to a plain
     * host/port target that was sent without one.
     */
    async function applyStoredSecrets(req, backend, config) {
        const secrets = await readSecretsFile(req);
        const merged = { ...(config || {}) };
        if (backend === 'qdrant') {
            if (secrets.qdrant_api_key && (merged.url || merged.apiKey)) merged.apiKey = secrets.qdrant_api_key;
        } else if (backend === 'milvus') {
            if (secrets.milvus_token) merged.token = secrets.milvus_token;
            if (secrets.milvus_password) merged.password = secrets.milvus_password;
            // The Milvus client takes one credential string: token, or user:password.
            if (!merged.token && merged.username && merged.password) {
                merged.token = `${merged.username}:${merged.password}`;
            }
        }
        return merged;
    }

    // A small, durable VectHare-owned purge intent. Standard's native purge
    // removes its initial-config file; this record survives a browser reload
    // so an interrupted ledger cleanup can be finished without guessing a
    // source/model or deleting any additional vectors.
    router.post('/identity-ledger/purge-recovery', requireVectHareCollection, async (req, res) => {
        let temporary = null;
        try {
            const { collectionId, action, record } = req.body;
            const root = path.join(req.user.directories.vectors, 'vecthare-owned-identity-ledgers');
            const filename = path.join(root, `purge-${createHash('sha256').update(collectionId).digest('hex')}.json`);
            const validate = value => {
                if (!value || value.version !== 1 || value.collectionId !== collectionId ||
                    !['standard', 'vectra'].includes(value.backend) ||
                    typeof value.source !== 'string' || !value.source || value.source.length > 256 ||
                    typeof value.modelFolder !== 'string' || value.modelFolder.length > 256 ||
                    value.destination !== JSON.stringify([value.backend, value.source, value.modelFolder]) ||
                    typeof value.originalRecordedAt !== 'string' ||
                    !Number.isFinite(Date.parse(value.originalRecordedAt)) ||
                    value.binding?.backend !== value.backend || value.binding?.source !== value.source) {
                    throw new Error('Invalid Standard purge recovery record');
                }
                identityLedgerPath(req, collectionId, value.binding);
            };
            const read = async () => {
                try { await fs.lstat(root); }
                catch (error) { if (error.code === 'ENOENT') return { status: 'missing' }; throw error; }
                await ensureIdentityLedgerDirectory(root);
                let stat;
                try { stat = await fs.lstat(filename); }
                catch (error) { if (error.code === 'ENOENT') return { status: 'missing' }; throw error; }
                if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 65536) return { status: 'invalid' };
                try {
                    const saved = JSON.parse(await fs.readFile(filename, 'utf8'));
                    validate(saved);
                    return { status: 'found', record: saved };
                } catch { return { status: 'invalid' }; }
            };
            if (action === 'read') return res.json({ success: true, ...await read() });
            if (!['prepare', 'clear'].includes(action)) throw new Error('Invalid recovery action');
            validate(record);
            const current = await read();
            if (current.status === 'invalid') throw new Error('Stored purge recovery record is invalid');
            if (current.status === 'found' && JSON.stringify(current.record) !== JSON.stringify(record)) {
                throw new Error('A different Standard destination has pending cleanup');
            }
            if (action === 'clear') {
                if (current.status === 'found') await fs.unlink(filename);
                return res.json({ success: true });
            }
            if (current.status === 'found') return res.json({ success: true });
            await ensureIdentityLedgerDirectory(root);
            const serialized = JSON.stringify(record);
            if (Buffer.byteLength(serialized, 'utf8') > 65536) throw new Error('Purge recovery record too large');
            temporary = path.join(root, `${path.basename(filename)}.${randomUUID()}.tmp`);
            await fs.writeFile(temporary, serialized, { flag: 'wx', encoding: 'utf8' });
            await fs.rename(temporary, filename);
            temporary = null;
            return res.json({ success: true });
        } catch (error) {
            return res.status(400).json({ error: error.message });
        } finally {
            if (temporary) await fs.unlink(temporary).catch(() => {});
        }
    });

    /** Store initial vectorization provenance alongside the backend collection.
     * This endpoint never changes vectors and never replaces a saved record.
     */
    router.post('/collections/initial-config', requireVectHareCollection, async (req, res) => {
        try {
            const { backend = 'standard', collectionId, physicalCollectionId,
                source = 'transformers', model = '', snapshot } = req.body || {};
            if (typeof collectionId !== 'string' || !collectionId ||
                !snapshot || typeof snapshot !== 'object' || Array.isArray(snapshot) ||
                snapshot.schemaVersion !== 1 || snapshot.storage?.backend !== backend ||
                snapshot.embedding?.provider !== source) {
                return res.status(400).json({ error: 'Invalid initial collection configuration' });
            }
            // 9d (P8): remote saved settings carry the SillyTavern user who wrote
            // them (set here, never by the client), so a per-collection store on a
            // server shared by several users has a known owner.
            const owner = (backend === 'qdrant' || backend === 'milvus') ? userHandle(req) : null;
            const record = { collectionId, snapshot, ...(owner ? { owner } : {}) };
            const json = JSON.stringify(record, null, 2);
            if (Buffer.byteLength(json, 'utf8') > 65536) return res.status(413).json({ error: 'Configuration too large' });

            let saved = false;
            // 9c: a v2 collection keeps its saved settings in its physical store,
            // keyed by its tenant (so two users' collections never share a slot).
            if (req.body?.storage !== undefined && req.body?.storage !== null) {
                const v = await resolveV2Request(req, backend, collectionId, req.body.storage);
                if (v.error) return res.status(v.status || 400).json({ error: v.error });
                if (!v.physical || (physicalCollectionId && physicalCollectionId !== v.physical)) {
                    return res.status(409).json({ error: 'The collection has no vectors in this storage yet' });
                }
                if (snapshot.storage?.layout !== v.layout) {
                    return res.status(400).json({ error: 'Snapshot storage layout does not match the request' });
                }
                const v2Record = { ...record, location: { layout: v.layout, physical: v.physical, dim: v.dim } };
                saved = backend === 'qdrant'
                    ? await qdrantBackend.saveInitialCollectionConfig(v.physical, v.configKey, v2Record)
                    : await milvusBackend.saveConfigV2(v.physical, v.configKey, v2Record);
                return res.json({ success: true, saved, location: v2Record.location });
            }
            if (backend === 'qdrant' || backend === 'milvus') {
                // Different logical collections can share one physical vector store.
                const key = `vecthare_initial_${createHash('sha256').update(collectionId).digest('hex')}`;
                if (backend === 'qdrant') {
                    if (physicalCollectionId !== collectionId &&
                        physicalCollectionId !== 'vecthare_multitenancy') {
                        return res.status(400).json({ error: 'Invalid physical Qdrant collection name' });
                    }
                    // VH-004/VH-033: nobody stamps a store that is not theirs.
                    const denied = physicalCollectionId === 'vecthare_multitenancy'
                        ? await legacyOwnerDenied(req, collectionId)
                        : await qdrantOwnerDenied(req, physicalCollectionId);
                    if (denied) return res.status(403).json({ error: denied });
                    saved = await qdrantBackend.saveInitialCollectionConfig(physicalCollectionId, key, record);
                    ownerCache.delete(ownerCacheKey(physicalCollectionId)); // its stamp is known now
                } else {
                    const denied = await legacyOwnerDenied(req, collectionId, `milvus:${collectionId}`);
                    if (denied) return res.status(403).json({ error: denied });
                    saved = await milvusBackend.saveInitialCollectionConfig(key, record);
                }
            } else if (['standard', 'vectra', 'lancedb'].includes(backend)) {
                const segment = (value) => {
                    if (typeof value !== 'string' || !value || value === '.' || value === '..' ||
                        !/^[a-zA-Z0-9_.-]+$/.test(value)) {
                        throw new Error('Unsafe collection storage path component');
                    }
                    return value;
                };
                let folder;
                if (backend === 'lancedb') {
                    await ensureLanceDBInitialized(req.user.directories);
                    // Use the same table-name conversion as lancedb-backend.js.
                    const table = collectionId.replace(/[^a-zA-Z0-9_.-]/g, '_');
                    const safeSource = segment(source);
                    if (!await (await lanceFor(req.user.directories)).getCollection(collectionId, source)) {
                        return res.status(409).json({ error: 'LanceDB table does not exist' });
                    }
                    folder = path.join(req.user.directories.vectors, 'lancedb', safeSource, `${segment(table)}.lance`);
                } else {
                    const parts = [req.user.directories.vectors, sanitize(source), sanitize(collectionId)];
                    // VH-094: next to the chunks, in SillyTavern's folder.
                    const scoped = await vectraScopedModel(req.user.directories, collectionId, source, model);
                    if (scoped) parts.push(sanitize(scoped));
                    folder = path.join(...parts);
                    // Do not call getIndex here: it creates missing indexes.
                    // Provenance must attach only to a real, written collection.
                    try {
                        if (!(await fs.stat(path.join(folder, 'index.json'))).isFile()) {
                            throw new Error('Vectra index missing');
                        }
                    } catch {
                        return res.status(409).json({ error: 'Vectra collection does not exist' });
                    }
                }
                if (!(await fs.stat(folder)).isDirectory()) {
                    return res.status(409).json({ error: 'Collection storage directory missing' });
                }
                const filename = path.join(folder, 'vecthare-initial-config.json');
                try {
                    // Exclusive creation prevents accidental overwrites.
                    await fs.writeFile(filename, json, { encoding: 'utf8', flag: 'wx' });
                    saved = true;
                } catch (error) {
                    if (error.code !== 'EEXIST') throw error;
                    const original = JSON.parse(await fs.readFile(filename, 'utf8'));
                    if (original.collectionId !== collectionId || original.snapshot?.schemaVersion !== 1) {
                        throw new Error('Existing collection configuration is invalid or belongs to another collection');
                    }
                }
            } else {
                return res.status(400).json({ error: 'Unsupported vector backend' });
            }
            res.json({ success: true, saved });
        } catch (error) {
            console.error(`[${pluginName}] Could not store initial collection configuration:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    /** Read saved provenance only. Never creates a collection or changes vectors. */
    // ========================================================================
    // EXPLICIT PHYSICAL EXISTENCE
    // ========================================================================
    // A registry entry, a folder name or a chunk count of zero is not evidence
    // that a collection exists. This asks the recorded backend directly and
    // answers 'present', 'absent' or 'unverified'. Anything that cannot be
    // established - unreachable backend, ambiguous destination, malformed
    // response - is 'unverified' and never reported as absent.
    router.post('/collections/exists', async (req, res) => {
        try {
            const { backend = 'vectra', collectionId, physicalCollectionId, source } = req.body;
            let { model } = req.body;
            // Read-only route: it reports whether a collection exists and
            // creates, changes and deletes nothing, so it is NOT restricted to
            // VectHare-owned identities. The ownership gate belongs on the
            // write and delete routes; applying it here left every other
            // collection permanently unverifiable. The identity is still
            // validated for shape so it can never escape its own destination.
            if (typeof collectionId !== 'string' || !collectionId || collectionId.length > 255 ||
                /[\\/]|\.\.|[\u0000-\u001f]/.test(collectionId)) {
                return res.status(400).json({ error: 'Invalid collectionId' });
            }
            const unverified = reason => res.json({ success: true, existence: 'unverified', reason });

            // 9c: v2 storage. Present = rows of this collection, or its live saved
            // settings; absent = neither (a purge leaves a tombstone, which is absent).
            if ((backend === 'qdrant' || backend === 'milvus') && req.body?.storage !== undefined && req.body?.storage !== null) {
                try {
                    const v = await resolveV2Request(req, backend, collectionId, req.body.storage);
                    if (v.error) return unverified(v.error);
                    if (!v.physical) return res.json({ success: true, existence: 'absent' });
                    const count = backend === 'qdrant'
                        ? (await qdrantBackend.collectionExists(v.physical) ? await qdrantBackend.countTenant(v.physical, v.tenant.tenantKey) : 0)
                        : await milvusBackend.countV2(v.physical, v.tenant.tenantKey);
                    if (count > 0) return res.json({ success: true, existence: 'present', count, countTrusted: true });
                    const config = backend === 'qdrant'
                        ? (await qdrantBackend.collectionExists(v.physical) ? await qdrantBackend.readInitialCollectionConfig(v.physical, v.configKey) : undefined)
                        : await milvusBackend.readConfigV2(v.physical, v.configKey);
                    return res.json({ success: true, existence: config === undefined ? 'absent' : 'present', count: 0, countTrusted: true });
                } catch (error) {
                    return unverified(`${backend} could not be checked: ${error.message}`);
                }
            }

            if (backend === 'qdrant') {
                const physical = physicalCollectionId || collectionId;
                // A shared multitenancy collection says nothing about whether
                // one logical collection inside it exists.
                if (physical !== collectionId) {
                    return unverified('A shared Qdrant collection cannot confirm one logical collection');
                }
                // VH-004: existence and counts of another user's store are not reported.
                const denied = await qdrantOwnerDenied(req, physical);
                if (denied) return unverified(denied);
                try {
                    const present = await qdrantBackend.collectionExists(physical);
                    if (!present) return res.json({ success: true, existence: 'absent' });
                    // P6: an exact count, so callers need not list every hash.
                    // A failed count leaves the collection present, count untrusted.
                    let counted = {};
                    try {
                        const countResponse = await qdrantBackend._request('POST',
                            `/collections/${encodeURIComponent(physical)}/points/count`, { exact: true });
                        const count = countResponse?.result?.count;
                        if (Number.isSafeInteger(count) && count >= 0) counted = { count, countTrusted: true };
                    } catch (error) {
                        console.debug(`[${pluginName}] Qdrant count failed for ${physical}: ${error.message}`);
                    }
                    return res.json({ success: true, existence: 'present', ...counted });
                } catch (error) {
                    return unverified(`Qdrant could not be checked: ${error.message}`);
                }
            }

            if (backend === 'milvus') {
                try {
                    if (!await milvusBackend.physicalCollectionExists()) {
                        return res.json({ success: true, existence: 'absent' });
                    }
                    // Milvus keeps every logical collection in one physical
                    // collection, so the count MUST be scoped to this logical
                    // collection (it used to count the whole store, which made
                    // every id read as present). The scope comes from the id.
                    const tenant = parseMilvusTenant(collectionId);
                    if (!tenant) {
                        return unverified('This collection id cannot be mapped to a Milvus collection identity');
                    }
                    const denied = await legacyOwnerDenied(req, collectionId, `milvus:${collectionId}`);
                    if (denied) return unverified(denied);
                    // VH-011: right after a delete, rows that had to stay because
                    // another collection stores the same text do not count: the
                    // collection is gone once nothing else of it remains.
                    if (req.body?.afterDelete === true) {
                        const { unshared, shared } = await milvusBackend.countUnsharedItems(tenant);
                        return unshared > 0
                            ? res.json({ success: true, existence: 'present', count: unshared + shared, countTrusted: true })
                            : res.json({ success: true, existence: 'absent', ...(shared ? { sharedLeftover: shared } : {}) });
                    }
                    const rows = await milvusBackend.countItems(tenant);
                    // A shared store cannot prove absence, so zero rows stays unverified.
                    return rows > 0
                        ? res.json({ success: true, existence: 'present', count: rows, countTrusted: true })
                        : unverified('A shared-store logical collection with no rows cannot be distinguished from an absent one');
                } catch (error) {
                    return unverified(`Milvus could not be checked: ${error.message}`);
                }
            }

            if (backend === 'lancedb') {
                try {
                    const lance = await lanceFor(req.user.directories);
                    const present = await lance.tableExists(collectionId, source);
                    if (!present) return res.json({ success: true, existence: 'absent' });
                    // P6: countRows() instead of listing every row.
                    let counted = {};
                    try {
                        const stats = await lance.getCollectionStats(collectionId, source);
                        const count = typeof stats?.count === 'bigint' ? Number(stats.count) : stats?.count;
                        if (Number.isSafeInteger(count) && count >= 0) counted = { count, countTrusted: true };
                    } catch (error) {
                        console.debug(`[${pluginName}] LanceDB count failed for ${collectionId}: ${error.message}`);
                    }
                    return res.json({ success: true, existence: 'present', ...counted });
                } catch (error) {
                    return unverified(`LanceDB could not be checked: ${error.message}`);
                }
            }

            // Vectra / Standard: the index directory for the exact recorded
            // provider and model folder. Never create anything here.
            // VH-094: the model folder is SillyTavern's for this source.
            const safeCollection = sanitize(String(collectionId));
            const safeSource = sanitize(String(source || ''));
            if (safeSource && safeSource === String(source) && safeCollection && (model || await stVectraModelScope(String(source), ''))) {
                // The folder name is the sanitized model (SillyTavern's names can contain "/").
                model = sanitize(await vectraScopedModel(req.user.directories, collectionId, String(source), model ? String(model) : ''));
            }
            const safeModel = sanitize(String(model || ''));
            if (!safeCollection || !safeSource || safeSource !== String(source) ||
                (model && safeModel !== String(model))) {
                return unverified('The recorded Vectra destination could not be resolved safely');
            }
            const base = path.join(req.user.directories.vectors, safeSource, safeCollection);
            const isIndex = async folder => {
                try {
                    if (!(await fs.stat(folder)).isDirectory()) return false;
                    return (await fs.stat(path.join(folder, 'index.json'))).isFile();
                } catch (error) {
                    if (error.code === 'ENOENT') return false;
                    throw error;
                }
            };
            try {
                if (model) {
                    const present = await isIndex(path.join(base, safeModel));
                    return res.json({ success: true, existence: present ? 'present' : 'absent' });
                }
                // No recorded model: the collection exists if its own folder is
                // an index, or if exactly one model subfolder is. Several
                // candidates leave the destination ambiguous.
                if (await isIndex(base)) return res.json({ success: true, existence: 'present' });
                let subfolders = [];
                try {
                    subfolders = (await fs.readdir(base, { withFileTypes: true }))
                        .filter(entry => entry.isDirectory()).map(entry => entry.name);
                } catch (error) {
                    if (error.code === 'ENOENT') return res.json({ success: true, existence: 'absent' });
                    throw error;
                }
                const present = [];
                for (const name of subfolders) {
                    if (await isIndex(path.join(base, name))) present.push(name);
                }
                if (present.length === 0) return res.json({ success: true, existence: 'absent' });
                if (present.length === 1) {
                    return res.json({ success: true, existence: 'present', model: present[0] });
                }
                return unverified(`${present.length} model folders exist for this collection; ` +
                    'the recorded destination is ambiguous');
            } catch (error) {
                return unverified(`The Vectra destination could not be read: ${error.message}`);
            }
        } catch (error) {
            console.error('[Similharity] Existence check failed:', error);
            return res.status(500).json({ error: error.message });
        }
    });

    router.post('/collections/initial-config/read', async (req, res) => {
        try {
            const { backend = 'standard', collectionId, physicalCollectionId,
                source = 'transformers', model = '' } = req.body || {};
            if (typeof collectionId !== 'string' || !collectionId ||
                typeof source !== 'string' || !source || typeof model !== 'string' ||
                !['standard', 'vectra', 'lancedb', 'qdrant', 'milvus'].includes(backend)) {
                return res.status(400).json({ error: 'Invalid collection location' });
            }
            let record;
            // 9c: v2 storage - read from the collection's own physical store,
            // or find it when its vector size is not known yet.
            if ((backend === 'qdrant' || backend === 'milvus') && req.body?.storage !== undefined && req.body?.storage !== null) {
                const v = await resolveV2Request(req, backend, collectionId, req.body.storage);
                if (v.error) return res.status(v.status || 400).json({ error: v.error });
                if (physicalCollectionId && v.physical && physicalCollectionId !== v.physical) {
                    return res.status(400).json({ error: 'Invalid physical collection name' });
                }
                if (!v.physical) return res.json({ success: true, status: 'missing' });
                let raw = backend === 'qdrant'
                    ? await qdrantBackend.readInitialCollectionConfig(v.physical, v.configKey)
                    : await milvusBackend.readConfigV2(v.physical, v.configKey);
                if (raw === undefined) return res.json({ success: true, status: 'missing', location: { layout: v.layout, physical: v.physical, dim: v.dim } });
                if (typeof raw === 'string') {
                    try { raw = JSON.parse(raw); } catch { return res.json({ success: true, status: 'invalid' }); }
                }
                return res.json({ success: true, status: 'found', record: raw,
                    location: { layout: v.layout, physical: v.physical, dim: v.dim } });
            }
            if (backend === 'qdrant' || backend === 'milvus') {
                const key = `vecthare_initial_${createHash('sha256').update(collectionId).digest('hex')}`;
                if (backend === 'qdrant') {
                    if (physicalCollectionId !== collectionId &&
                        physicalCollectionId !== 'vecthare_multitenancy') {
                        return res.status(400).json({ error: 'Invalid physical Qdrant collection name' });
                    }
                    const denied = physicalCollectionId === 'vecthare_multitenancy'
                        ? await legacyOwnerDenied(req, collectionId)
                        : await qdrantOwnerDenied(req, physicalCollectionId);
                    record = await qdrantBackend.readInitialCollectionConfig(physicalCollectionId, key);
                    // A record that does not exist reveals nothing; one that does is the owner's.
                    if (denied && record !== undefined) return res.status(403).json({ error: denied });
                } else {
                    if (physicalCollectionId !== 'vecthare_main') {
                        return res.status(400).json({ error: 'Invalid physical Milvus collection name' });
                    }
                    const denied = await legacyOwnerDenied(req, collectionId, `milvus:${collectionId}`);
                    record = await milvusBackend.readInitialCollectionConfig(key);
                    if (denied && record !== undefined) return res.status(403).json({ error: denied });
                    if (record !== undefined) {
                        try { record = JSON.parse(record); }
                        catch { return res.json({ success: true, status: 'invalid' }); }
                    }
                }
            } else {
                // Match the existing write path, but do not use getIndex/getCollection:
                // those accessors can create directories or indexes.
                let folders = [];
                const folderSources = new Map();
                // A saved model may belong to a provider other than the one
                // selected globally now. Search existing user-owned folders only,
                // and refuse ambiguous matches below. No index is created.
                const roots = async base => {
                    try { return (await fs.readdir(base, { withFileTypes: true }))
                        .filter(entry => entry.isDirectory()).map(entry => entry.name); }
                    catch (error) { if (error.code === 'ENOENT') return []; throw error; }
                };
                if (backend === 'lancedb') {
                    if (source !== '*' && (source === '.' || source === '..' ||
                        !/^[a-zA-Z0-9_.-]+$/.test(source))) {
                        return res.status(400).json({ error: 'Unsafe source' });
                    }
                    const table = collectionId.replace(/[^a-zA-Z0-9_.-]/g, '_');
                    if (!table || table === '.' || table === '..') {
                        return res.status(400).json({ error: 'Invalid table name' });
                    }
                    const lanceRoot = path.join(req.user.directories.vectors, 'lancedb');
                    const sources = source === '*' ? await roots(lanceRoot) : [source];
                    folders = sources.filter(src => /^[a-zA-Z0-9_.-]+$/.test(src) &&
                        src !== '.' && src !== '..')
                        .map(src => {
                            const folder = path.join(lanceRoot, src, `${table}.lance`);
                            folderSources.set(folder, src);
                            return folder;
                        });
                } else {
                    const safeCollection = sanitize(collectionId);
                    if (!safeCollection || (model && !sanitize(model)) ||
                        (source !== '*' && (!source || sanitize(source) !== source))) {
                        return res.status(400).json({ error: 'Invalid collection location' });
                    }
                    // VH-094: the saved settings sit next to the chunks, in SillyTavern's folder.
                    const vectraModel = source === '*' ? model
                        : await vectraScopedModel(req.user.directories, collectionId, source, model);
                    const safeModel = sanitize(vectraModel);
                    const vectorsRoot = req.user.directories.vectors;
                    const sources = source === '*' ? (await roots(vectorsRoot)).filter(src => src !== 'lancedb')
                        : [source];
                    for (const src of sources) {
                        const safeSource = sanitize(src);
                        if (!safeSource || safeSource !== src) continue;
                        const base = path.join(vectorsRoot, safeSource, safeCollection);
                        if (vectraModel) {
                            const folder = path.join(base, safeModel);
                            folderSources.set(folder, src);
                            folders.push(folder);
                        } else {
                            const models = await roots(base);
                            for (const folder of [base, ...models.map(name => path.join(base, name))]) {
                                folderSources.set(folder, src);
                                folders.push(folder);
                            }
                        }
                    }
                }
                let existing = false;
                let existingCount = 0;
                const records = [];
                for (const folder of folders) {
                    try {
                        if (!(await fs.stat(folder)).isDirectory()) continue;
                        if (backend !== 'lancedb' &&
                            !(await fs.stat(path.join(folder, 'index.json'))).isFile()) continue;
                    } catch (error) {
                        if (error.code === 'ENOENT') continue;
                        throw error;
                    }
                    existing = true;
                    existingCount++;
                    let json;
                    try { json = await fs.readFile(path.join(folder, 'vecthare-initial-config.json'), 'utf8'); }
                    catch (error) {
                        if (error.code === 'ENOENT') continue;
                        throw error;
                    }
                    if (Buffer.byteLength(json, 'utf8') > 65536) {
                        return res.json({ success: true, status: 'invalid' });
                    }
                    let parsedRecord;
                    try { parsedRecord = JSON.parse(json); }
                    catch { return res.json({ success: true, status: 'invalid' }); }
                    // The record must describe the physical provider folder it
                    // came from, including when a wildcard location was requested.
                    if (parsedRecord?.snapshot?.embedding?.provider !== folderSources.get(folder)) {
                        return res.json({ success: true, status: 'invalid' });
                    }
                    records.push(parsedRecord);
                }
                if (!existing) return res.status(404).json({ error: 'Collection not found' });
                // Several physical Vectra indexes mean the model location is ambiguous,
                // even if only one currently has a provenance file.
                if ((!model && existingCount > 1) || records.length > 1) {
                    return res.json({ success: true, status: 'ambiguous' });
                }
                record = records[0];
            }
            if (record === undefined) return res.json({ success: true, status: 'missing' });
            return res.json({ success: true, status: 'found', record });
        } catch (error) {
            console.error(`[${pluginName}] Could not read initial collection configuration:`, error);
            // VH-080: say why, so "unavailable" does not hide a refused API key.
            // A backend that could not connect (e.g. after a restart with a wrong
            // key) answers "not initialized"; its last connection attempt says why.
            const last = lastInitFailure[String(req.body?.backend || '')];
            const notConnected = /not initialized|not connected/i.test(String(error?.message || ''));
            const backendStatus = Number(error?.status) || (notConnected ? Number(last?.status) : NaN);
            const credentials = backendStatus === 401 || backendStatus === 403 ||
                /unauthenticated|authentication|invalid api key|permission denied|access denied/i.test(
                    String(error?.message || '') + (notConnected ? ` ${last?.message || ''}` : ''));
            res.status(503).json({ error: 'Collection configuration unavailable',
                ...(Number.isInteger(backendStatus) ? { backendStatus } : {}),
                ...(credentials ? { credentials: true } : {}),
                backend: String(req.body?.backend || '') });
        }
    });

    /**
     * POST /api/plugins/similharity/storage/move
     * Step 9d / P6: { action: 'copy', backend, collectionId, from, to, embedding }
     *            or { action: 'delete-copy', backend, collectionId, location }.
     * Same backend only. The extension holds the collection's lease and commits
     * the result; this route only moves or removes vectors (see storage-move.js).
     */
    router.post('/storage/move', requireVectHareCollection, async (req, res) => {
        try {
            const { action, backend, collectionId } = req.body || {};
            if (backend !== 'qdrant' && backend !== 'milvus') return res.status(400).json({ error: 'Moves are offered for Qdrant and Milvus only' });
            if (backend === 'qdrant' && !qdrantBackend.baseUrl) return res.status(503).json({ error: 'Qdrant not initialized' });
            if (backend === 'milvus' && !milvusBackend.isConnected) return res.status(503).json({ error: 'Milvus not initialized' });
            const tenant = tenantFor(req, collectionId);
            if (!tenant) return res.status(400).json({ error: 'This collection id or user cannot be scoped' });
            if (backend === 'qdrant') {
                const denied = await qdrantOwnerDenied(req, collectionId);
                if (denied) return res.status(403).json({ error: denied });
            }
            // VH-002/VH-033: moving or deleting out of an older shared store needs evidence too.
            const touchesShared = backend === 'milvus' ||
                [req.body?.from?.layout, req.body?.location?.layout].includes('legacy-mt');
            if (touchesShared) {
                const denied = await legacyOwnerDenied(req, collectionId, `${backend}:${collectionId}`);
                if (denied) return res.status(403).json({ error: denied });
            }
            const ctx = { qdrant: qdrantBackend, milvus: milvusBackend };
            if (action === 'copy') {
                const result = await copyCollection(ctx, { backend, collectionId, tenant,
                    fromRaw: req.body.from, toRaw: req.body.to,
                    embedding: { source: typeof req.body.embedding?.source === 'string' ? req.body.embedding.source.slice(0, 64) : undefined,
                        model: typeof req.body.embedding?.model === 'string' ? req.body.embedding.model.slice(0, 128) : undefined } });
                return res.json({ success: true, ...result });
            }
            if (action === 'delete-copy') {
                // Fail closed: never delete the place VectHare's metadata file
                // says the collection is now (a stale tab, a wrong button).
                const current = await readOwnedStateFile(ownedStatePath(req).filename);
                if (current.status !== 'found') {
                    return res.status(409).json({ error: `VectHare's metadata file could not be read (${current.status}); nothing was deleted` });
                }
                const locator = collectRecordedCollections(current.state).get(collectionId)?.locator;
                const loc = req.body.location;
                if (!locator) {
                    return res.status(409).json({ error: 'Where this collection is stored now is not recorded; nothing was deleted.' });
                }
                // VH-005: compare what would really be deleted (the normalised location),
                // not the raw request; a physical name that the layout ignores is refused.
                let target;
                try { target = parseLocation(backend, collectionId, tenant.tenantKey, loc); }
                catch (error) { return res.status(400).json({ error: error.message }); }
                if (loc?.physical != null && loc.physical !== target.physical) {
                    return res.status(400).json({ error: 'The location names a different store than its layout; nothing was deleted.' });
                }
                let currentTarget = null;
                if (locator.backend === backend) {
                    try { currentTarget = parseLocation(backend, collectionId, tenant.tenantKey, locator); } catch { currentTarget = null; }
                }
                const isCurrent = locator.backend === backend && (
                    (currentTarget && currentTarget.layout === target.layout && currentTarget.physical === target.physical) ||
                    (locator.layout === target.layout && locator.physical === target.physical) ||
                    (!currentTarget && locator.layout === target.layout)); // unparseable record: fail closed per layout
                if (isCurrent) {
                    return res.status(409).json({ error: 'That is where the collection is stored now; it is not an old copy. Nothing was deleted.' });
                }
                return res.json({ success: true, ...await deleteCopy(ctx, { backend, collectionId, tenant, locationRaw: loc }) });
            }
            return res.status(400).json({ error: 'Unknown action' });
        } catch (error) {
            console.error(`[${pluginName}] storage/move error:`, error);
            res.status([400, 403, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/storage/report
     * Step 9a / P1: where every collection VectHare's metadata file knows about
     * is physically stored, how many vectors it holds and where its saved
     * original settings were found. READ-ONLY (see storage-report.js).
     * Body: { defaults: { backend, source, model, qdrantMultitenancy } } - today's
     * settings, only used to say what today's settings-based rule would pick.
     */
    router.post('/storage/report', async (req, res) => {
        try {
            const vectorsDir = req.user.directories.vectors;
            const report = await buildStorageReport({
                vectorsDir,
                dataRoot: path.dirname(req.user.directories.root ?? path.dirname(vectorsDir)),
                isAdmin: req.user?.profile?.admin === true,
                defaults: req.body?.defaults || {},
                readOwnedState: async () => {
                    const current = await readOwnedStateFile(ownedStatePath(req).filename);
                    return current.status === 'found'
                        ? { status: 'found', state: current.state, token: current.token }
                        : { status: current.status, token: current.token };
                },
                qdrant: qdrantBackend.baseUrl ? qdrantBackend : null,
                milvus: milvusBackend.isConnected ? milvusBackend : null,
                lance: () => lanceFor(req.user.directories),
                sanitize,
                parseMilvusTenant,
                ALL_TENANTS,
                // 9c: this user's tenant key of a logical id (v2 layouts), and
                // the per-collection Milvus store name it implies.
                tenantKeyOf: id => tenantFor(req, id)?.tenantKey ?? null,
                milvusOwnName,
                userHandle: userHandle(req),
            });
            res.json(report);
        } catch (error) {
            console.error(`[${pluginName}] storage/report error:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    /** GET /api/plugins/similharity/collections — list collections in all backends. */
    router.get('/collections', async (req, res) => {
        try {
            const vectorsPath = req.user.directories.vectors;
            // 9d (P8): on a server with several users, physical Qdrant stores are
            // listed only to their owner (by the stamp in their saved settings);
            // unstamped (older) ones only to an admin.
            const restrict = req.user?.profile?.admin !== true && await isMultiUserServer(req);
            // VH-003: an owner that cannot be read hides the store (never shows it);
            // VH-004: an unstamped store is listed to its owner by evidence.
            const handle = userHandle(req);
            // An owner that cannot be read throws: the store is hidden and the
            // Qdrant listing is reported incomplete (VH-008).
            const allowed = async name => {
                const owner = await qdrantOwnerOf(name);
                if (owner !== undefined) return owner === handle;
                return !await legacyOwnerDenied(req, name, `qdrant:${name}`);
            };
            const scan = await scanAllSourcesForCollections(vectorsPath,
                restrict ? { ownerOf: qdrantOwnerOf, handle, allowed } : null);
            // VH-085: this user's collections inside the shared v2 stores (Qdrant
            // or Milvus multitenancy, new layout) and Milvus per-collection
            // stores, with the store they live in, so they can be found again
            // without collections.json (e.g. after Start fresh).
            const v2 = await listV2TenantCollections(req);
            scan.collections.push(...v2.collections);
            for (const backend of v2.incomplete) if (!scan.incomplete.includes(backend)) scan.incomplete.push(backend);

            res.json({
                success: true,
                count: scan.collections.length,
                collections: scan.collections,
                // VH-008: backends whose listing may be missing collections.
                inventoryReport: 1,
                incompleteBackends: scan.incomplete,
            });
        } catch (error) {
            console.error(`[${pluginName}] collections error:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    /**
     * GET /api/plugins/similharity/sources
     * Lists available embedding sources
     */
    router.get('/sources', async (req, res) => {
        try {
            const vectorsPath = req.user.directories.vectors;
            const entries = await fs.readdir(vectorsPath, { withFileTypes: true });
            const sources = entries.filter(e => e.isDirectory()).map(e => e.name);

            res.json({ success: true, sources });
        } catch (error) {
            console.error(`[${pluginName}] sources error:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/get-embedding
     * Get embedding for single text
     */
    router.post('/get-embedding', async (req, res) => {
        try {
            const { text, source, model = '', isQuery = false } = req.body;

            if (!text || !source) {
                return res.status(400).json({ error: 'text and source are required' });
            }

            // #14: the caller says whether this is a search query or a stored
            // passage; asymmetric providers (Cohere) need the distinction.
            const embedding = await getEmbeddingForSource(source, text, model, req.user.directories, req,
                isQuery === true);
            res.json({ success: true, embedding });

        } catch (error) {
            console.error(`[${pluginName}] get-embedding error:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/batch-embeddings
     * Get embeddings for multiple texts
     */
    router.post('/batch-embeddings', async (req, res) => {
        try {
            const { texts, source, model = '' } = req.body;

            if (!texts || !Array.isArray(texts) || !source) {
                return res.status(400).json({ error: 'texts array and source are required' });
            }

            const embeddings = await getVectorsForSource(source, texts, model, req.user.directories, req);
            res.json({ success: true, embeddings });

        } catch (error) {
            console.error(`[${pluginName}] batch-embeddings error:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    // ========================================================================
    // BACKEND MANAGEMENT ENDPOINTS
    // ========================================================================

    /**
     * GET /api/plugins/similharity/backend/health/:backend
     * Health check for specific backend
     */
    router.get('/backend/health/:backend', async (req, res) => {
        try {
            const { backend } = req.params;
            let healthy = false;
            let message = '';

            switch (backend) {
                case 'vectra':
                case 'standard':
                    healthy = true;
                    message = 'Vectra is always available (file-based)';
                    break;

                case 'lancedb':
                    healthy = (await lanceFor(req.user.directories)).basePath != null;
                    message = healthy ? 'LanceDB initialized' : 'LanceDB not initialized';
                    break;

                case 'qdrant':
                    healthy = await qdrantBackend.healthCheck();
                    message = healthy ? 'Qdrant connected' : 'Qdrant not available';
                    if (healthy && qdrantBackend.supportsCollectionMetadata() === false) message = qdrantBackend.tooOldMessage();
                    break;

                case 'milvus':
                    healthy = await milvusBackend.healthCheck();
                    message = healthy ? 'Milvus connected' : 'Milvus not available';
                    // the server version, read once per connection (shown in VectHare)
                    if (healthy && !milvusBackend.v2Version) await milvusBackend.assertV2Supported().catch(() => {});
                    break;

                default:
                    return res.status(400).json({ error: `Unknown backend: ${backend}` });
            }

            res.json({ backend, healthy, message,
                ...(backend === 'qdrant' ? { version: qdrantBackend.serverVersion || null, writable: qdrantBackend.supportsCollectionMetadata() !== false } : {}),
                ...(backend === 'milvus' ? { version: milvusBackend.v2Version?.version ?? null, writable: milvusBackend.v2Version ? milvusBackend.v2Version.ok : null } : {}) });

        } catch (error) {
            console.error(`[${pluginName}] backend/health error:`, error);
            res.status(500).json({ backend: req.params.backend, healthy: false, error: error.message });
        }
    });

    // VH-001: one Qdrant/Milvus connection serves every user of this SillyTavern
    // server. On a multi-user server only an admin may set it; other users get the
    // admin's connection (their own address settings are not used). The admin's
    // last working target (no keys; the key is read from the admin's key store)
    // is remembered so the connection can come back after a server restart even
    // when a non-admin is the first to use it.
    const SHARED_BACKENDS = new Set(['qdrant', 'milvus']);
    const serverStateFile = req => path.join(
        path.dirname(req.user.directories.root ?? path.dirname(req.user.directories.vectors)),
        '_vecthare-server', 'backend-connections.json');
    const backendFor = name => name === 'qdrant' ? qdrantBackend : milvusBackend;
    const isBackendReady = name => name === 'qdrant' ? Boolean(qdrantBackend.baseUrl) : Boolean(milvusBackend.isConnected && milvusBackend.client);

    async function readServerConnections(req) {
        try {
            const parsed = JSON.parse(await fs.readFile(serverStateFile(req), 'utf8'));
            return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
        } catch {
            return {};
        }
    }

    async function rememberServerConnection(req, backend, config) {
        const target = backend === 'qdrant'
            ? { url: config.url || null, host: config.host || null, port: config.port || null }
            : { address: config.address || null, ssl: config.ssl === true, username: config.username || null, dimensions: config.dimensions ?? null };
        const record = { target, keyStore: secretsPath(req).filename, by: req.user?.profile?.handle || null, at: new Date().toISOString() };
        const file = serverStateFile(req);
        try {
            const all = await readServerConnections(req);
            all[backend] = record;
            await fs.mkdir(path.dirname(file), { recursive: true });
            const temporary = `${file}.${randomUUID()}.tmp`;
            await fs.writeFile(temporary, JSON.stringify(all, null, 2), { encoding: 'utf8', mode: 0o600 });
            await fs.rename(temporary, file);
        } catch (error) {
            console.warn(`[${pluginName}] Could not remember the ${backend} connection: ${error.message}`);
        }
    }

    async function restoreServerConnection(req, backend) {
        const record = (await readServerConnections(req))[backend];
        if (!record?.target) return false;
        let secrets = {};
        try {
            const parsed = JSON.parse(await fs.readFile(record.keyStore, 'utf8'));
            if (parsed && typeof parsed === 'object') secrets = parsed;
        } catch { /* no key store: connect without a key */ }
        const config = { ...record.target };
        if (backend === 'qdrant') {
            if (typeof secrets.qdrant_api_key === 'string' && secrets.qdrant_api_key && config.url) config.apiKey = secrets.qdrant_api_key;
        } else if (typeof secrets.milvus_token === 'string' && secrets.milvus_token) {
            config.token = secrets.milvus_token;
        } else if (config.username && typeof secrets.milvus_password === 'string' && secrets.milvus_password) {
            config.token = `${config.username}:${secrets.milvus_password}`;
        }
        await backendFor(backend).initialize(config);
        console.log(`[${pluginName}] ${backend} connection restored from the admin's last setup`);
        return true;
    }

    const sharedBackendRestricted = async req => req.user?.profile?.admin !== true && await isMultiUserServer(req);

    /**
     * POST /api/plugins/similharity/backend/init/:backend
     * Initialize specific backend
     */
    // VH-080: why the last connection attempt per backend failed (cleared on success).
    const lastInitFailure = {};

    router.post('/backend/init/:backend', async (req, res) => {
        try {
            const { backend } = req.params;
            const config = req.body;

            if (SHARED_BACKENDS.has(backend) && await sharedBackendRestricted(req)) {
                if (!isBackendReady(backend)) {
                    try {
                        await restoreServerConnection(req, backend);
                    } catch (error) {
                        return res.status(503).json({
                            error: `The ${backend} connection set up by the server admin is not reachable: ${error.message}`,
                            code: 'admin_connection_unavailable',
                        });
                    }
                }
                if (!isBackendReady(backend)) {
                    return res.status(403).json({
                        error: `On a multi-user SillyTavern server only an admin can set up the ${backend === 'qdrant' ? 'Qdrant' : 'Milvus'} connection. Ask an admin to open SillyTavern with VectHare once.`,
                        code: 'admin_required',
                    });
                }
                return res.json({ success: true, managedByAdmin: true, message: `${backend} connection is managed by the server admin`,
                    ...(backend === 'qdrant' && qdrantBackend.supportsCollectionMetadata() === false ? { warning: qdrantBackend.tooOldMessage() } : {}) });
            }

            switch (backend) {
                case 'vectra':
                case 'standard':
                    res.json({ success: true, message: 'Vectra requires no initialization' });
                    break;

                case 'lancedb':
                    await lanceFor(req.user.directories);
                    res.json({ success: true, message: 'LanceDB initialized' });
                    break;

                case 'qdrant':
                    await qdrantBackend.initialize(await applyStoredSecrets(req, 'qdrant', config));
                    await rememberServerConnection(req, 'qdrant', config || {});
                    delete lastInitFailure.qdrant; // VH-080
                    res.json({ success: true, message: 'Qdrant initialized', version: qdrantBackend.serverVersion || null,
                        ...(qdrantBackend.supportsCollectionMetadata() === false ? { warning: qdrantBackend.tooOldMessage() } : {}) });
                    break;

                case 'milvus':
                    await milvusBackend.initialize(await applyStoredSecrets(req, 'milvus', config));
                    await rememberServerConnection(req, 'milvus', config || {});
                    delete lastInitFailure.milvus; // VH-080
                    res.json({ success: true, message: 'Milvus initialized' });
                    break;

                default:
                    return res.status(400).json({ error: `Unknown backend: ${backend}` });
            }

        } catch (error) {
            console.error(`[${pluginName}] backend/init error:`, error);
            // VH-080: remembered, so later "not initialized" answers can say why.
            lastInitFailure[String(req.params.backend || '')] = { status: Number(error?.status) || null, message: String(error?.message || '') };
            res.status(500).json({ error: error.message });
        }
    });

/**
 * U10: outbound URL guard for fetches the plugin makes itself (BananaBread and
 * KoboldCpp embeddings, /rerank). It matters only when SillyTavern has more
 * than one user: then any logged-in user could point the server at loopback,
 * LAN or cloud-metadata addresses. With a single user nothing changes. The
 * admin can allow private targets with SIMILHARITY_ALLOW_PRIVATE_URLS=1 in the
 * server's environment.
 *
 * Best effort: the address is resolved here and again by fetch, so a DNS name
 * that changes between the two is not caught.
 */
let multiUserCache = { at: 0, value: false };

async function isMultiUserServer(req) {
    const now = Date.now();
    if (now - multiUserCache.at < 60_000) return multiUserCache.value;
    let value = true; // fail closed: if the user list cannot be read, treat as multi-user
    try {
        const dataRoot = path.dirname(req.user.directories.root ?? path.dirname(req.user.directories.vectors));
        const entries = await fs.readdir(dataRoot, { withFileTypes: true });
        let users = 0;
        for (const entry of entries) {
            if (!entry.isDirectory() || entry.name.startsWith('_') || entry.name.startsWith('.')) continue;
            const hasSettings = await fs.access(path.join(dataRoot, entry.name, 'settings.json')).then(() => true, () => false);
            if (hasSettings) users++;
        }
        value = users > 1;
    } catch {
        value = true;
    }
    multiUserCache = { at: now, value };
    return value;
}

/** VH-066: the eight 16-bit groups of an IPv6 address, or null. */
function ipv6Groups(address) {
    let text = String(address).toLowerCase().split('%')[0];
    const dotted = /(\d+\.\d+\.\d+\.\d+)$/.exec(text);
    if (dotted) {
        const parts = dotted[1].split('.').map(Number);
        if (parts.some(n => n > 255)) return null;
        text = text.slice(0, -dotted[1].length) + ((parts[0] << 8) | parts[1]).toString(16) + ':' + ((parts[2] << 8) | parts[3]).toString(16);
    }
    const halves = text.split('::');
    if (halves.length > 2) return null;
    const head = halves[0] ? halves[0].split(':') : [];
    const tail = halves.length === 2 && halves[1] ? halves[1].split(':') : [];
    const fill = halves.length === 2 ? 8 - head.length - tail.length : 0;
    const groups = [...head, ...Array(Math.max(fill, 0)).fill('0'), ...tail].map(g => parseInt(g || '0', 16));
    return groups.length === 8 && groups.every(g => Number.isInteger(g) && g >= 0 && g <= 0xffff) ? groups : null;
}

function isPrivateAddress(address) {
    let ip = address;
    // VH-066: IPv4 inside IPv6 (::ffff:a.b.c.d in any spelling, ::a.b.c.d, 64:ff9b::/96)
    // is judged as the IPv4 address it reaches.
    if (net.isIPv6(ip)) {
        const g = ipv6Groups(ip);
        if (!g) return true;
        const embedded = (g[0] === 0 && g[1] === 0 && g[2] === 0 && g[3] === 0 && g[4] === 0 && (g[5] === 0xffff || g[5] === 0) &&
            !(g[5] === 0 && g[6] === 0 && g[7] <= 1)) || (g[0] === 0x64 && g[1] === 0xff9b && g.slice(2, 6).every(x => x === 0));
        if (embedded) ip = `${g[6] >> 8}.${g[6] & 255}.${g[7] >> 8}.${g[7] & 255}`;
    }
    if (net.isIPv4(ip)) {
        const [a, b] = ip.split('.').map(Number);
        return a === 0 || a === 10 || a === 127 || (a === 169 && b === 254) ||
            (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) ||
            (a === 100 && b >= 64 && b <= 127) || a >= 224;
    }
    if (net.isIPv6(ip)) {
        const lower = ip.toLowerCase();
        return lower === '::' || lower === '::1' || lower.startsWith('fe8') || lower.startsWith('fe9') ||
            lower.startsWith('fea') || lower.startsWith('feb') || lower.startsWith('fc') || lower.startsWith('fd');
    }
    return true; // not an IP we understand
}

/**
 * Throws if `url` must not be fetched on behalf of this request.
 * @param {object} req
 * @param {URL} url
 */
async function assertOutboundUrlAllowed(req, url) {
    if (!(await isMultiUserServer(req))) return {};
    if (['1', 'true', 'yes'].includes(String(process.env.SIMILHARITY_ALLOW_PRIVATE_URLS || '').toLowerCase())) return {};
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
        throw new Error('Blocked: only http(s) URLs are allowed on a multi-user server');
    }
    const host = url.hostname.replace(/^\[|\]$/g, '');
    let addresses;
    if (net.isIP(host)) {
        addresses = [host];
    } else {
        try {
            addresses = (await dns.lookup(host, { all: true })).map(entry => entry.address);
        } catch {
            throw new Error(`Blocked: could not resolve "${host}"`);
        }
    }
    if (!addresses.length || addresses.some(isPrivateAddress)) {
        throw new Error('Blocked: this server has several SillyTavern users, so loopback/private-network URLs are not allowed. ' +
            'The server admin can allow them by setting SIMILHARITY_ALLOW_PRIVATE_URLS=1.');
    }
    // VH-036: a checked URL must not be redirected somewhere unchecked.
    return { redirect: 'manual' };
}

function parseProviderUrl(label, apiUrl) {
    try { return new URL(String(apiUrl)); } catch (e) { throw new Error(`${label}: Invalid URL format "${apiUrl}" - ${e.message}`); }
}

/** VH-036: a guarded request answered with a redirect is refused. */
function refuseRedirect(response) {
    if (response.type === 'opaqueredirect' || (response.status >= 300 && response.status < 400)) {
        throw new Error('Blocked: the embedding server answered with a redirect, which is not followed on a multi-user server');
    }
    return response;
}

/**
 * Get multiple embeddings for texts from specified source
 */
async function getVectorsForSource(source, texts, model, directories, req, isQuery = false) {
    // Specialized batch handling
    if (source === 'bananabread') {
        // BananaBread llama.cpp-compatible endpoint
        let apiUrl = req.body.apiUrl || 'http://localhost:8008';
        let apiKey = req.body.apiKey || '';

        if (!apiUrl || typeof apiUrl !== 'string' || apiUrl.trim() === '') {
            throw new Error('BananaBread: apiUrl is missing or invalid. Configure the embedding URL in VectHare settings.');
        }
        apiUrl = apiUrl.trim();

        let url;
        try {
            url = new URL(apiUrl);
            url.pathname = '/embedding';
        } catch (e) {
            throw new Error(`BananaBread: Invalid URL format "${apiUrl}" - ${e.message}`);
        }
        const guard = await assertOutboundUrlAllowed(req, url);

        const headers = { 'Content-Type': 'application/json' };
        if (apiKey) {
            headers['Authorization'] = `Bearer ${apiKey}`;
        }

        const BATCH_SIZE = 20;
        const allEmbeddings = [];

        try {
            for (let i = 0; i < texts.length; i += BATCH_SIZE) {
                const batchTexts = texts.slice(i, i + BATCH_SIZE);
                // console.log(`[BananaBread] Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(texts.length / BATCH_SIZE)} (${batchTexts.length} items)`);

                const response = refuseRedirect(await fetch(url.toString(), {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({ content: batchTexts }),
                    signal: AbortSignal.timeout(120_000), // VH-019: never wait forever
                    ...guard,
                }));

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`BananaBread: ${response.statusText} ${errorText}`);
                }

                const data = await response.json();
                const embeddings = data.embedding;

                if (!embeddings) {
                    throw new Error('BananaBread: Invalid response format (missing embedding)');
                }

                if (Array.isArray(embeddings)) {
                    // Check for single item flattened response
                    if (batchTexts.length === 1 && typeof embeddings[0] === 'number') {
                        allEmbeddings.push(embeddings);
                    } else if (Array.isArray(embeddings[0])) {
                        allEmbeddings.push(...embeddings);
                    } else {
                         // Some providers might return a single vector for batch of 1 as a flat array even if asked in batch mode
                         if (batchTexts.length === 1 && Array.isArray(embeddings)) {
                             allEmbeddings.push(embeddings);
                         } else {
                            throw new Error('BananaBread: Unexpected embedding format');
                         }
                    }
                } else {
                    throw new Error('BananaBread: Unexpected embedding format (not array)');
                }
            }

            return allEmbeddings;

        } catch (e) {
            console.error(`[BananaBread] Batch embedding error:`, e);
            throw e;
        }
    }

    // Default fallback: sequential processing for other sources
    const results = [];
    for (const text of texts) {
        results.push(await _getLegacySingleEmbedding(source, text, model, directories, req, isQuery));
    }
    return results;
}

/**
 * Wrapper for single embedding (backwards compatibility)
 * `isQuery` matters for providers with asymmetric embeddings (Cohere): a stored
 * passage and a search query must not be embedded the same way (#14).
 */
async function getEmbeddingForSource(source, text, model, directories, req, isQuery = false) {
    const vectors = await getVectorsForSource(source, [text], model, directories, req, isQuery);
    return vectors[0];
}

/**
 * Helper function for KoboldCpp embedding generation
 * @param {string} text - Text to embed
 * @param {string} model - Model name
 * @param {object} req - Request object containing apiUrl
 * @returns {Promise<number[]>} Embedding vector
 */
async function _getKoboldCppEmbedding(text, model, req) {
    const apiUrl = req.body?.apiUrl || 'http://localhost:5001';

    let url;
    try {
        url = new URL(apiUrl);
        // Ensure we're hitting the embeddings endpoint
        if (!url.pathname.includes('/embeddings')) {
            url.pathname = url.pathname.replace(/\/?$/, '/v1/embeddings').replace(/\/+/g, '/');
        }
    } catch (e) {
        throw new Error(`KoboldCpp: Invalid URL format "${apiUrl}" - ${e.message}`);
    }
    const guard = await assertOutboundUrlAllowed(req, url);

    const response = refuseRedirect(await fetch(url.toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(120_000), // VH-019: never wait forever
        ...guard,
        body: JSON.stringify({
            input: text,
            model: model || 'koboldcpp',
        }),
    }));

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`KoboldCpp: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    // OpenAI format: { data: [{ embedding: [...] }] }
    if (data?.data?.[0]?.embedding) {
        return data.data[0].embedding;
    }
    // Fallback: direct embedding array
    if (Array.isArray(data?.embedding)) {
        return data.embedding;
    }
    throw new Error('KoboldCpp: Invalid response format - no embedding found');
}

/**
 * Legacy single embedding function (renamed)
 */
async function _getLegacySingleEmbedding(source, text, model, directories, req, isQuery = false) {
    switch (source) {
        case 'transformers': {
            const { getTransformersVector } = await import('../../src/vectors/embedding.js');
            return await getTransformersVector(text);
        }
        case 'openai':
        case 'togetherai':
        case 'mistral':
        case 'electronhub':
        case 'openrouter': {
            const { getOpenAIVector } = await import('../../src/vectors/openai-vectors.js');
            // ElectronHub requires a model name, provide default if empty
            const effectiveModel = (source === 'electronhub' && !model) ? 'text-embedding-ada-002' : model;
            return await getOpenAIVector(text, source, directories, effectiveModel);
        }
        case 'nomicai': {
            const { getNomicAIVector } = await import('../../src/vectors/nomicai-vectors.js');
            return await getNomicAIVector(text, source, directories);
        }
        case 'cohere': {
            // #14: Cohere needs search_query for queries and search_document for
            // stored passages. This was hard-coded to the query form, so every
            // inserted chunk was embedded as if it were a search query.
            const { getCohereVector } = await import('../../src/vectors/cohere-vectors.js');
            return await getCohereVector(text, isQuery === true, directories, model);
        }
        case 'koboldcpp': {
            return await _getKoboldCppEmbedding(text, model, req);
        }
        case 'ollama': {
            const { getOllamaVector } = await import('../../src/vectors/ollama-vectors.js');
            const apiUrl = req.body?.apiUrl || 'http://localhost:11434';
            await assertOutboundUrlAllowed(req, parseProviderUrl('Ollama', apiUrl)); // VH-036
            const keep = req.body?.keep || false;
            return await getOllamaVector(text, apiUrl, model, keep, directories);
        }
        case 'llamacpp': {
            const { getLlamaCppVector } = await import('../../src/vectors/llamacpp-vectors.js');
            const apiUrl = req.body?.apiUrl || 'http://localhost:8080';
            await assertOutboundUrlAllowed(req, parseProviderUrl('llama.cpp', apiUrl)); // VH-036
            // SillyTavern after 1.19.1 adds a model argument: (text, apiUrl, model,
            // directories); 1.19.1 and older take (text, apiUrl, directories).
            return await llamaCppTakesModel()
                ? await getLlamaCppVector(text, apiUrl, model ? String(model) : '', directories)
                : await getLlamaCppVector(text, apiUrl, directories);
        }
        case 'bananabread': {
            // Legacy single-item fallback (should normally be handled by batch handler)
            return (await getVectorsForSource(source, [text], model, directories, req))[0];
        }
        case 'vllm': {
            const { getVllmVector } = await import('../../src/vectors/vllm-vectors.js');
            const apiUrl = req.body?.apiUrl || 'http://localhost:8000';
            await assertOutboundUrlAllowed(req, parseProviderUrl('vLLM', apiUrl)); // VH-036
            return await getVllmVector(text, apiUrl, model, directories);
        }
        case 'palm':
        case 'vertexai': {
            const googleVectors = await import('../../src/vectors/google-vectors.js');
            if (source === 'palm') {
                return await googleVectors.getMakerSuiteVector(text, model, req);
            } else {
                return await googleVectors.getVertexVector(text, model, req);
            }
        }
        case 'extras': {
            const { getExtrasVector } = await import('../../src/vectors/extras-vectors.js');
            const extrasUrl = req.body?.extrasUrl || 'http://localhost:5100';
            await assertOutboundUrlAllowed(req, parseProviderUrl('Extras', extrasUrl)); // VH-036 (Pass 19)
            const extrasKey = req.body?.extrasKey || '';
            return await getExtrasVector(text, extrasUrl, extrasKey);
        }
        default:
            throw new Error(`Unknown vector source: ${source}`);
    }
}

    // ========================================================================
    // UNIFIED CHUNK ENDPOINTS
    // ========================================================================

    /**
     * POST /api/plugins/similharity/chunks/list
     * List all chunks in a collection with pagination
     * Body: { backend, collectionId, source?, model?, offset?, limit?, includeVectors?, filters? }
     */
    router.post('/chunks/list', async (req, res) => {
        try {
            const {
                backend = 'vectra',
                collectionId,
                source = 'transformers',
                model = '',
                offset = 0,
                limit = 100,
                includeVectors = false,
                hashesOnly = false,
                strictRead = false,
                allHashes = false,
                filters = {}
            } = req.body;

            if (!collectionId) {
                return res.status(400).json({ error: 'collectionId is required' });
            }

            // Auto-init LanceDB if needed
            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            // VH-027: allHashes asks for every hash of the collection, with
            // `complete: true` only when the reply is known to hold all of them.
            const wantAll = allHashes === true;
            const result = await handler.list(collectionId, source, model, req.user.directories, {
                offset: wantAll ? 0 : offset,
                limit: wantAll ? Number.MAX_SAFE_INTEGER : limit,
                includeVectors: wantAll ? false : includeVectors,
                hashesOnly: wantAll || hashesOnly,
                strictRead: wantAll || strictRead === true,
                allHashes: wantAll,
                filters
            });
            if (wantAll) {
                result.items = (result.items || []).map(item => ({ hash: item.hash }));
                result.complete = result.complete === true ||
                    (result.hasMore === false && Number(result.total) === result.items.length);
            }

            res.json({
                success: true,
                backend: handler.type,
                collectionId,
                ...result
            });

        } catch (error) {
            console.error(`[${pluginName}] chunks/list error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/chunks/get-by-hashes
     * P5: the stored records of one collection for the given hashes (every
     * physical copy), for verification after a write or delete without sending
     * the whole collection to the browser.
     * Body: { backend, collectionId, hashes: number[], source?, model?, filters? }
     * Reply: { success, complete: true, items: [{hash, text, metadata}] }
     * Milvus: scoped `hash in [...]` (scope from the middleware); other backends
     * read their collection completely on the server and filter.
     */
    router.post('/chunks/get-by-hashes', async (req, res) => {
        try {
            const { backend = 'vectra', collectionId, source = 'transformers', model = '', filters = {} } = req.body || {};
            if (!collectionId) return res.status(400).json({ error: 'collectionId is required' });
            const wanted = toSafeHashList(req.body?.hashes);
            if (backend === 'lancedb') await ensureLanceDBInitialized(req.user.directories);
            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            let items;
            if (handler.v2) {
                items = await handler.getByHashes(wanted);
            } else if (handler.type === 'milvus') {
                items = await milvusBackend.getItemsByHashes(wanted, filters);
            } else {
                const listed = await handler.list(collectionId, source, model, req.user.directories, {
                    offset: 0, limit: Number.MAX_SAFE_INTEGER, includeVectors: false,
                    strictRead: true, filters,
                });
                if (!Array.isArray(listed?.items) || listed.hasMore === true) {
                    throw new Error('The collection could not be read completely');
                }
                const set = new Set(wanted.map(String));
                items = listed.items.filter(item => set.has(String(item?.hash)))
                    .map(item => ({ hash: item.hash, text: item.text, metadata: item.metadata }));
            }
            res.json({ success: true, complete: true, backend: handler.type, collectionId, items });
        } catch (error) {
            console.error(`[${pluginName}] chunks/get-by-hashes error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * GET /api/plugins/similharity/chunks/:hash
     * Get single chunk by hash
     * Query: backend, collectionId, source?, model?
     */
    router.get('/chunks/:hash', async (req, res) => {
        try {
            const { hash } = req.params;
            const { backend = 'vectra', collectionId, source = 'transformers', model = '' } = req.query;
            // Parse filters from query string (JSON encoded)
            let filters = {};
            if (req.query.filters) {
                try {
                    filters = JSON.parse(req.query.filters);
                } catch (e) {
                    console.warn(`[${pluginName}] Invalid filters JSON:`, req.query.filters);
                }
            }
            // Milvus: the validated, server-derived scope replaces client filters.
            if (req.vecthareMilvusFilters) filters = req.vecthareMilvusFilters;

            if (!collectionId) {
                return res.status(400).json({ error: 'collectionId is required' });
            }

            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            const chunk = await handler.get(collectionId, hash, source, model, req.user.directories, filters);

            if (!chunk) {
                return res.status(404).json({ error: 'Chunk not found' });
            }

            res.json({ success: true, chunk });

        } catch (error) {
            console.error(`[${pluginName}] chunks/:hash error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/chunks/insert
     * Insert new chunks
     * Body: { backend, collectionId, items: [{hash, text, index, metadata?, vector?}], source?, model? }
     */
    router.post('/chunks/insert', requireVectHareCollection, async (req, res) => {
        try {
            const {
                backend = 'vectra',
                collectionId,
                items,
                source = 'transformers',
                model = '',
                filters = {}
            } = req.body;

            if (!collectionId) {
                return res.status(400).json({ error: 'collectionId is required' });
            }
            if (!items || !Array.isArray(items)) {
                return res.status(400).json({ error: 'items array is required' });
            }

            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            // The endpoint also accepts direct plugin callers (not just the
            // VectHare extension). Preserve supplied provenance exactly; only
            // otherwise stamp a clearly *unverified* API-origin record.
            // Never change hashes, vectors, text or grouping.
            const submitted = !Array.isArray(items) ? items : items.map(item => {
                if (item?.metadata?.vecthareProvenance != null) return item;
                const meta = item?.metadata || {};
                const provenance = {
                    version: 1,
                    writer: 'api-insert',
                    recordKind: 'api-insert',
                    sourceType: meta.contentType || (meta.isScene ? 'scene' : 'unknown'),
                    collectionId,
                    chunkUid: `chunk_${randomUUID()}`,
                    chunkUidStatus: 'provisional',
                    groupingStrategy: meta.strategy || 'unknown',
                    groupingVersion: 'legacy',
                    sourceIdentity: { status: 'unverified', messageUids: [] },
                };
                if (meta.chunkIndex != null) provenance.segmentIndex = meta.chunkIndex;
                else if (item.index != null) provenance.segmentIndex = item.index;
                if (meta.entryUid != null) provenance.entryUid = meta.entryUid;
                if (meta.sourceName != null) provenance.sourceLabel = meta.sourceName;
                if (typeof item.text === 'string') {
                    provenance.submittedTextFingerprint = 'sha256:' + createHash('sha256')
                        .update(item.text).digest('hex');
                }
                return { ...item, metadata: { ...meta, vecthareProvenance: provenance } };
            });

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            const location = await handler.insert(collectionId, submitted, source, model, req.user.directories, req, filters);

            res.json({
                success: true,
                backend: handler.type,
                collectionId,
                inserted: items.length,
                // 9c: where a v2 collection's vectors went (the extension records it).
                ...(handler.v2 && location ? { storage: location } : {}),
            });

        } catch (error) {
            console.error(`[${pluginName}] chunks/insert error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * PATCH /api/plugins/similharity/chunks/:hash/text
     * Update chunk text (triggers re-embedding)
     * Body: { backend, collectionId, text, source?, model? }
     */
    router.patch('/chunks/:hash/text', requireVectHareCollection, async (req, res) => {
        try {
            const { hash } = req.params;
            const {
                backend = 'vectra',
                collectionId,
                text,
                source = 'transformers',
                model = '',
                filters = {}
            } = req.body;

            if (!collectionId || !text) {
                return res.status(400).json({ error: 'collectionId and text are required' });
            }

            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            const result = await handler.updateText(collectionId, hash, text, source, model, req.user.directories, req, filters);

            res.json({
                success: true,
                backend: handler.type,
                ...result
            });

        } catch (error) {
            console.error(`[${pluginName}] chunks/:hash/text error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * PATCH /api/plugins/similharity/chunks/:hash/metadata
     * Update chunk metadata (no re-embedding)
     * Body: { backend, collectionId, metadata, source?, model? }
     */
    router.patch('/chunks/:hash/metadata', requireVectHareCollection, async (req, res) => {
        try {
            const { hash } = req.params;
            const {
                backend = 'vectra',
                collectionId,
                metadata,
                source = 'transformers',
                model = '',
                filters = {}
            } = req.body;

            if (!collectionId || !metadata) {
                return res.status(400).json({ error: 'collectionId and metadata are required' });
            }

            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            const result = await handler.updateMetadata(collectionId, hash, metadata, source, model, req.user.directories, filters);

            res.json({
                success: true,
                backend: handler.type,
                ...result
            });

        } catch (error) {
            console.error(`[${pluginName}] chunks/:hash/metadata error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/chunks/delete
     * Delete chunks by hash
     * Body: { backend, collectionId, hashes, source?, model? }
     */
    router.post('/chunks/delete', requireVectHareCollection, async (req, res) => {
        try {
            const {
                backend = 'vectra',
                collectionId,
                hashes,
                source = 'transformers',
                model = '',
                filters = {}
            } = req.body;

            if (!collectionId) {
                return res.status(400).json({ error: 'collectionId is required' });
            }
            if (!hashes || !Array.isArray(hashes)) {
                return res.status(400).json({ error: 'hashes array is required' });
            }

            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            const deleted = await handler.delete(collectionId, hashes, source, model, req.user.directories, filters);

            res.json({
                success: true,
                backend: handler.type,
                collectionId,
                deleted
            });

        } catch (error) {
            console.error(`[${pluginName}] chunks/delete error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/chunks/query
     * Query chunks by semantic similarity
     * Body: { backend, collectionId, queryVector OR searchText, topK?, threshold?, source?, model?, includeVectors? }
     */
    router.post('/chunks/query', async (req, res) => {
        try {
            const {
                backend = 'vectra',
                collectionId,
                queryVector,
                searchText,
                topK = 10,
                threshold = 0.0,
                source = 'transformers',
                model = '',
                includeVectors = false,
                filters = {}
            } = req.body;

            if (!collectionId) {
                return res.status(400).json({ error: 'collectionId is required' });
            }
            if (!queryVector && !searchText) {
                return res.status(400).json({ error: 'queryVector or searchText is required' });
            }

            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            // Get query vector if not provided
            let vector = queryVector;
            if (!vector && searchText) {
                vector = await getEmbeddingForSource(source, searchText, model, req.user.directories, req, true);
            }

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            const results = await handler.query(collectionId, vector, topK, threshold, source, model, req.user.directories, {
                includeVectors,
                filters
            });

            res.json({
                success: true,
                backend: handler.type,
                collectionId,
                count: results.length,
                results
            });

        } catch (error) {
            console.error(`[${pluginName}] chunks/query error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/chunks/hybrid-query
     * Hybrid search combining vector similarity and keyword matching
     * Body: { backend, collectionId, searchText OR queryVector, keywords?, topK?, options?, filters?, source?, model? }
     */
    router.post('/chunks/hybrid-query', async (req, res) => {
        try {
            const {
                backend = 'qdrant',
                collectionId,
                queryVector,
                searchText,
                keywords,
                topK = 10,
                options = {},
                filters = {},
                source = 'transformers',
                model = '',
                hybridOptions = {}
            } = req.body;

            if (!collectionId) {
                return res.status(400).json({ error: 'collectionId is required' });
            }
            if (!queryVector && !searchText) {
                return res.status(400).json({ error: 'queryVector or searchText is required' });
            }

            // Only Qdrant, Milvus support native hybrid query
            if (backend !== 'qdrant' && backend !== 'milvus') {
                return res.status(400).json({ error: `Backend ${backend} does not support native hybrid query` });
            }
            // 9b fix: the Milvus backend has no hybridQuery; this route used to
            // embed the query and then fail with a TypeError (500) on every RAG
            // query. Say so up front; the extension falls back to vector search.
            if (backend === 'milvus' && typeof milvusBackend.hybridQuery !== 'function') {
                return res.status(400).json({ error: 'Milvus has no native hybrid query here; use vector search' });
            }

            // Generate embedding if searchText provided
            let vector = queryVector;
            if (!vector && searchText) {
                vector = await getEmbeddingForSource(source, searchText, model, req.user.directories, req, true);
            }

            // Extract keywords if not provided
            let extractedKeywords = keywords;
            if (!extractedKeywords && searchText) {
                // Simple keyword extraction: split by spaces, remove short words
                extractedKeywords = searchText
                    .toLowerCase()
                    .split(/\s+/)
                    .filter(word => word.length > 2)
                    .slice(0, 10); // Limit to 10 keywords
            }

            // Merge hybridOptions with options
            const mergedOptions = {
                ...hybridOptions,
                ...options
            };

            if (req.vh2) {
                const handler = getV2Handler(req);
                if (!handler.hybrid) {
                    return res.status(400).json({ error: 'Milvus has no native hybrid query here; use vector search' });
                }
                const results = await handler.hybrid(vector, extractedKeywords || [], topK, mergedOptions);
                return res.json({
                    success: true, backend, collectionId, count: results.length,
                    results: results.map(r => ({ hash: r.hash, text: r.text, score: r.score, metadata: r.metadata,
                        vectorScore: r.debug?.vectorScore, textScore: r.debug?.keywordScore, debug: r.debug })),
                });
            }

            if (backend === 'qdrant') {
                const results = await qdrantBackend.hybridQuery(
                    collectionId,
                    vector,
                    extractedKeywords || [],
                    topK,
                    mergedOptions,
                    filters
                );

                res.json({
                    success: true,
                    backend: 'qdrant',
                    collectionId,
                    count: results.length,
                    results: results.map(r => ({
                        hash: r.hash,
                        text: r.text,
                        score: r.score,
                        metadata: r.metadata,
                        vectorScore: r.debug?.vectorScore,
                        textScore: r.debug?.keywordScore,
                        debug: r.debug
                    }))
                });
            } else if (backend === 'milvus') {
                // Milvus hybrid query implementation (if available)
                const results = await milvusBackend.hybridQuery(
                    collectionId,
                    vector,
                    extractedKeywords || [],
                    topK,
                    mergedOptions,
                    filters
                );

                res.json({
                    success: true,
                    backend: 'milvus',
                    collectionId,
                    count: results.length,
                    results: results.map(r => ({
                        hash: r.hash,
                        text: r.text,
                        score: r.score,
                        metadata: r.metadata,
                        vectorScore: r.debug?.vectorScore,
                        textScore: r.debug?.keywordScore,
                        debug: r.debug
                    }))
                });
            }

        } catch (error) {
            console.error(`[${pluginName}] chunks/hybrid-query error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/chunks/purge
     * Purge all chunks in a collection
     * Body: { backend, collectionId, source?, model? }
     */
    router.post('/chunks/purge', async (req, res) => {
        try {
            const {
                backend = 'vectra',
                collectionId,
                source = 'transformers',
                model = '',
                filters = {},
                confirmUnowned = false
            } = req.body;

            if (!collectionId) {
                return res.status(400).json({ error: 'collectionId is required' });
            }
            // The ownership gate stands for every caller EXCEPT a whole-collection
            // deletion the user explicitly confirmed for this exact destination.
            // A collection created before VectHare recorded ownership carries no
            // marker and would otherwise be undeletable forever. A filtered purge
            // never qualifies: it edits a collection rather than removing it.
            if (!isVectHareCollection(collectionId)) {
                const wholeCollection = !filters || !Object.keys(filters).length;
                if (confirmUnowned !== true || !wholeCollection) {
                    return res.status(403).json({
                        error: 'Only VectHare-owned collection IDs may be modified',
                    });
                }
                console.warn(`[${pluginName}] Purging ${backend}/${source} collection ` +
                    `"${collectionId}" on explicit user confirmation: not a VectHare-owned ID`);
            }

            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            const purged = await handler.purge(collectionId, source, model, req.user.directories, filters);
            // Purge invalidates only VectHare-owned identity records. No ST
            // chat file, settings file or vanilla vector index is touched.
            // Only a whole-collection purge drops the ledger. A filtered purge leaves the
            // collection (and its saved identity marker) in place, so its ledger must stay;
            // stale `indexed` entries are re-verified against the backend before use.
            if (!filters || !Object.keys(filters).length) {
                const ledgerRoot = path.join(req.user.directories.vectors, 'vecthare-owned-identity-ledgers');
                try { await fs.lstat(ledgerRoot); await ensureIdentityLedgerDirectory(ledgerRoot); }
                catch (error) { if (error.code !== 'ENOENT') throw error; }
                const files = await fs.readdir(ledgerRoot).catch(error => {
                    if (error.code === 'ENOENT') return [];
                    throw error;
                });
                for (const name of files) {
                    if (!/^[a-f0-9]{64}\.json$/.test(name)) continue;
                    const filename = path.join(ledgerRoot, name);
                    const stored = await fs.readFile(filename, 'utf8').then(JSON.parse).catch(() => null);
                    if (stored?.collectionId === collectionId &&
                        stored.binding?.backend === backend &&
                        stored.binding?.source === source &&
                        stored.binding?.model === model) await fs.unlink(filename);
                }
            }

            res.json({
                success: true,
                backend: handler.type,
                collectionId,
                message: `Collection ${collectionId} purged`,
                // VH-011: Milvus' older shared store - rows whose text another
                // collection also stores, left in place.
                ...(Number(purged?.leftShared) > 0 ? { leftShared: Number(purged.leftShared) } : {}),
            });

        } catch (error) {
            console.error(`[${pluginName}] chunks/purge error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    /** The shared vectors directory also contains SillyTavern-owned data. */
    router.post('/purge-all', (_req, res) => {
        res.status(403).json({ error: 'Global vector purge is unsafe; purge verified VectHare collections individually' });
    });

    /**
     * POST /api/plugins/similharity/rerank
     * Rerank documents using BananaBread's reranking endpoint
     * Body: { apiUrl, query, documents, top_k? }
     */
    router.post('/rerank', async (req, res) => {
        try {
            const {
                apiUrl = 'http://localhost:8008',
                apiKey = '',
                query,
                documents,
                top_k = 10,
                task_description
            } = req.body;

            if (!query || !documents || !Array.isArray(documents)) {
                return res.status(400).json({ error: 'query and documents array required' });
            }

            const url = new URL(apiUrl);
            url.pathname = '/v1/rerank';
            const guard = await assertOutboundUrlAllowed(req, url);

            // Default task description if not provided
            const finalTaskDescription = task_description || "Given the following document from a role play, which of the following documents are most relevant to it?";

            const headers = { 'Content-Type': 'application/json' };
            if (apiKey) {
                headers['Authorization'] = `Bearer ${apiKey}`;
            }

            const response = refuseRedirect(await fetch(url.toString(), {
                method: 'POST',
                headers: headers,
                ...guard,
                signal: AbortSignal.timeout(60_000), // VH-019
                body: JSON.stringify({
                    query,
                    documents,
                    top_k,
                    return_documents: false,
                    task_description: finalTaskDescription
                }),
            }));

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`BananaBread rerank failed: ${response.statusText} ${errorText}`);
            }

            const data = await response.json();
            res.json(data);
        } catch (error) {
            console.error(`[${pluginName}] rerank error:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    /**
     * POST /api/plugins/similharity/chunks/stats
     * Get collection statistics
     * Body: { backend, collectionId, source?, model? }
     */
    router.post('/chunks/stats', async (req, res) => {
        try {
            const {
                backend = 'vectra',
                collectionId,
                source = 'transformers',
                model = '',
                filters = {}
            } = req.body;

            if (!collectionId) {
                return res.status(400).json({ error: 'collectionId is required' });
            }

            if (backend === 'lancedb') {
                await ensureLanceDBInitialized(req.user.directories);
            }

            const handler = req.vh2 ? getV2Handler(req) : getBackendHandler(backend);
            const stats = await handler.stats(collectionId, source, model, req.user.directories, filters);

            res.json({
                success: true,
                backend: handler.type,
                collectionId,
                stats
            });

        } catch (error) {
            console.error(`[${pluginName}] chunks/stats error:`, error);
            res.status([400, 409].includes(error.status) ? error.status : 500).json({ error: error.message });
        }
    });

    console.log(`[${pluginName}] Plugin initialized successfully`);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get vectra index for a collection
 */
/**
 * VH-094: the model folder SillyTavern itself uses for a Vectra (Standard)
 * collection - src/endpoints/vectors.js getSourceSettings() -> getModelScope().
 * VectHare reads Standard collections through SillyTavern's own
 * /api/vector/list|query|delete, so the plugin must write to the same folder.
 * It used the model VectHare sent instead: for transformers (SillyTavern takes
 * the model from config.yaml) and fixed-model sources that was a different
 * folder, so nothing written through the plugin could be read back.
 */
let llamaCppModelArg = null;
/** Whether this SillyTavern's llama.cpp embedding function takes a model argument. */
async function llamaCppTakesModel() {
    if (llamaCppModelArg === null) {
        try {
            const { getLlamaCppVector } = await import('../../src/vectors/llamacpp-vectors.js');
            llamaCppModelArg = typeof getLlamaCppVector === 'function' && getLlamaCppVector.length >= 4;
        } catch {
            llamaCppModelArg = false;
        }
    }
    return llamaCppModelArg;
}

async function stVectraModelScope(source, model) {
    const m = model ? String(model) : '';
    switch (source) {
        case 'transformers': {
            try {
                const { getConfigValue } = await import('../../src/util.js');
                return String(getConfigValue('extensions.models.embedding', '') || '');
            } catch {
                return m;
            }
        }
        // SillyTavern after 1.19.1 keeps a llama.cpp model (when one is given)
        // as the folder, like other sources; 1.19.1 and older never do.
        case 'llamacpp': return (await llamaCppTakesModel()) ? m : '';
        case 'extras': return '';
        case 'mistral': return 'mistral-embed';
        case 'nomicai': return 'nomic-embed-text-v1.5';
        case 'electronhub': return m || 'text-embedding-3-small';
        case 'openrouter': return m || 'openai/text-embedding-3-large';
        case 'palm':
        case 'vertexai': return m || 'text-embedding-005';
        case 'chutes': return m || 'chutes-qwen-qwen3-embedding-8b';
        case 'nanogpt': return m || 'text-embedding-3-small';
        case 'siliconflow': return m || 'Qwen/Qwen3-Embedding-0.6B';
        case 'workers_ai': return m || '@cf/baai/bge-m3';
        default: return m;
    }
}

const vectraScopeMoves = new Map();

/**
 * VH-094: the model folder to use for a Vectra collection - SillyTavern's own
 * (see stVectraModelScope). Chunks and VectHare's saved-settings file written by
 * earlier plugin builds under the model VectHare sent are moved there once.
 * @returns {Promise<string>} the model folder name to use ('' = no subfolder)
 */
async function vectraScopedModel(directories, collectionId, source, model) {
    const scope = await stVectraModelScope(source, model);
    const raw = model ? String(model) : '';
    if (sanitize(scope) === sanitize(raw) || !source || !collectionId) return scope;
    const collectionDir = path.join(directories.vectors, sanitize(source), sanitize(collectionId));
    const oldPath = raw ? path.join(collectionDir, sanitize(raw)) : collectionDir;
    const newPath = scope ? path.join(collectionDir, sanitize(scope)) : collectionDir;
    const key = `${oldPath}\u0000${newPath}`;
    if (!vectraScopeMoves.has(key)) {
        vectraScopeMoves.set(key, moveVectraScope(oldPath, newPath).catch((error) => {
            vectraScopeMoves.delete(key);
            console.warn(`[similharity] Could not move Standard collection ${collectionId} to SillyTavern's folder:`, error?.message || error);
        }));
    }
    await vectraScopeMoves.get(key);
    return scope;
}

async function getIndex(directories, collectionId, source, model) {
    const scope = await vectraScopedModel(directories, collectionId, source, model);
    const pathToFile = scope
        ? path.join(directories.vectors, sanitize(source), sanitize(collectionId), sanitize(scope))
        : path.join(directories.vectors, sanitize(source), sanitize(collectionId));

    const store = new vectra.LocalIndex(pathToFile);

    if (!await store.isIndexCreated()) {
        await store.createIndex();
    }

    return store;
}

/**
 * VH-094: scan one user's Vectra folders for VectHare collections whose index
 * sits in the collection folder itself while SillyTavern reads a model
 * subfolder for that source, and move them (see vectraScopedModel).
 */
async function moveOldVectraScopes(directories) {
    let sources;
    try {
        sources = (await fs.readdir(directories.vectors, { withFileTypes: true }))
            .filter(entry => entry.isDirectory() && entry.name !== 'lancedb' && !entry.name.startsWith('vecthare-owned'))
            .map(entry => entry.name);
    } catch (error) {
        if (error.code === 'ENOENT') return;
        throw error;
    }
    for (const source of sources) {
        if (!await stVectraModelScope(source, '')) continue;
        let collections;
        try {
            collections = (await fs.readdir(path.join(directories.vectors, source), { withFileTypes: true }))
                .filter(entry => entry.isDirectory() && entry.name.startsWith('vecthare_')).map(entry => entry.name);
        } catch { continue; }
        for (const collectionId of collections) {
            try {
                await fs.access(path.join(directories.vectors, source, collectionId, 'index.json'));
            } catch { continue; }
            await vectraScopedModel(directories, collectionId, source, '');
        }
    }
}

/** VH-094: move an index (and VectHare's saved-settings file) from the old plugin folder into SillyTavern's. */
async function moveVectraScope(oldPath, newPath) {
    const oldIndex = new vectra.LocalIndex(oldPath);
    const oldConfig = path.join(oldPath, 'vecthare-initial-config.json');
    const newConfig = path.join(newPath, 'vecthare-initial-config.json');
    const exists = async file => { try { return (await fs.stat(file)).isFile(); } catch { return false; } };
    const oldItems = await oldIndex.isIndexCreated() ? await oldIndex.listItems() : [];
    const newIndex = new vectra.LocalIndex(newPath);
    const newItems = await newIndex.isIndexCreated() ? await newIndex.listItems() : [];
    if (oldItems.length && newItems.length) {
        console.warn(`[similharity] Standard collection has chunks in both ${oldPath} and ${newPath}; leaving both as they are.`);
        return;
    }
    if (oldItems.length) {
        if (!await newIndex.isIndexCreated()) await newIndex.createIndex();
        const seen = new Set();
        await newIndex.beginUpdate();
        for (const item of oldItems) {
            const hash = String(item.metadata?.hash);
            if (seen.has(hash)) continue;
            seen.add(hash);
            await newIndex.upsertItem({ vector: item.vector, metadata: item.metadata });
        }
        await newIndex.endUpdate();
        // Only the old index file goes; the folder may be SillyTavern's own parent folder.
        await fs.rm(path.join(oldPath, 'index.json'), { force: true });
        console.log(`[similharity] Moved Standard collection chunks (${seen.size} unique of ${oldItems.length}) into SillyTavern's folder ${newPath}`);
    }
    if (await exists(oldConfig) && !await exists(newConfig)) {
        await fs.mkdir(newPath, { recursive: true });
        await fs.rename(oldConfig, newConfig);
    }
}

/**
 * Simple string hash function
 */
/**
 * VH-034: the same 53-bit hash SillyTavern's getStringHash() computes (cyrb53),
 * which VectHare uses for every chunk. The plugin used a different 32-bit hash
 * here, so a text edit through the plugin filed the chunk under a hash nothing
 * else computes.
 */
function getStringHash(str, seed = 0) {
    if (typeof str !== 'string') return 0;
    let h1 = 0xdeadbeef ^ seed;
    let h2 = 0x41c6ce57 ^ seed;
    for (let i = 0, ch; i < str.length; i++) {
        ch = str.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, 2654435761);
        h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

/**
 * Scan all sources for collections
 */
async function scanAllSourcesForCollections(vectorsPath, ownerFilter = null) {
    const allCollections = [];
    // VH-008: backends whose listing may be missing collections (a read failed,
    // or the listing was filtered). The extension never prunes those.
    const incomplete = new Set();

    try {
        // Scan vectra indexes
        const vectraIndexes = await findAllIndexes(vectorsPath, '', () => incomplete.add('vectra'));

        const vectraCollectionMap = new Map();
        for (const index of vectraIndexes) {
            const key = `${index.source}:${index.collectionId}`;

            if (!vectraCollectionMap.has(key)) {
                vectraCollectionMap.set(key, {
                    id: index.collectionId,
                    source: index.source,
                    backend: 'vectra',
                    indexes: [],
                    totalChunks: 0
                });
            }

            let chunkCount = await getChunkCountFromIndex(index.indexPath);
            if (chunkCount === null) { incomplete.add('vectra'); chunkCount = 0; }
            vectraCollectionMap.get(key).indexes.push({
                modelPath: index.modelPath,
                chunkCount: chunkCount
            });
            vectraCollectionMap.get(key).totalChunks += chunkCount;
        }

        for (const [key, collection] of vectraCollectionMap) {
            if (!collection.id) continue;

            const primaryIndex = collection.indexes.reduce((best, curr) =>
                curr.chunkCount > best.chunkCount ? curr : best
            , collection.indexes[0]);

            const models = collection.indexes.map(idx => ({
                name: idx.modelPath || '(default)',
                path: idx.modelPath,
                chunkCount: idx.chunkCount
            }));

            allCollections.push({
                id: collection.id,
                source: collection.source,
                backend: 'vectra',
                chunkCount: collection.totalChunks,
                modelCount: collection.indexes.length,
                model: primaryIndex?.modelPath || '',
                models: models
            });
        }

        // Scan LanceDB
        const lancedbPath = path.join(vectorsPath, 'lancedb');
        let lancedbFolder = false;
        try {
            await fs.access(lancedbPath);
            lancedbFolder = true;
            const lancedbInstance = await lanceFor({ vectors: vectorsPath });

            const sourceDirs = await fs.readdir(lancedbPath, { withFileTypes: true });
            const sources = sourceDirs.filter(d => d.isDirectory() && !d.name.endsWith('.lance')).map(d => d.name);

            for (const source of sources) {
                try {
                    const db = await lancedbInstance.getDatabase(source);
                    const tableNames = await db.tableNames();

                    for (const tableName of tableNames) {
                        if (!tableName) continue;
                        try {
                            const table = await db.openTable(tableName);
                            const count = await table.countRows();

                            allCollections.push({
                                id: tableName,
                                source: source,
                                backend: 'lancedb',
                                chunkCount: count,
                                modelCount: 1
                            });
                        } catch (e) {
                            incomplete.add('lancedb');
                            console.warn(`[${pluginName}] LanceDB: Failed to open table '${tableName}' in source '${source}':`, e.message);
                        }
                    }
                } catch (e) {
                    incomplete.add('lancedb');
                    console.warn(`[${pluginName}] LanceDB: Failed to scan source '${source}':`, e.message);
                }
            }
        } catch (e) {
            // LanceDB folder doesn't exist - this is normal if LanceDB hasn't been used
            if (lancedbFolder) incomplete.add('lancedb');
            console.debug(`[${pluginName}] LanceDB folder not found or not accessible:`, e.message);
        }

        // Scan Qdrant (uses REST API, so check if initialized via baseUrl)
        // VH-008: each store is read on its own; a failure, an unreachable server
        // or an owner check that could not be made marks the listing incomplete.
        try {
            if (!qdrantBackend.baseUrl || !await qdrantBackend.healthCheck()) {
                incomplete.add('qdrant');
            } else {
                const listing = await qdrantBackend._request('GET', '/collections');
                const collections = listing.result?.collections?.map(c => c.name);
                if (!Array.isArray(collections)) throw new Error('Qdrant did not list its collections');
                for (const collectionName of collections) {
                    // 9c: a v2 shared store is not a collection of its own
                    // (its collections are tenants inside it).
                    if (MT_V2_NAME.test(collectionName)) continue;
                    try {
                        if (ownerFilter && (isReservedSharedCollection(collectionName) ||
                            !await ownerFilter.allowed(collectionName))) continue;
                        const name = qdrantBackend._parseCollectionName(collectionName);
                        const route = `/collections/${name}`;
                        const countResponse = await qdrantBackend._request(
                            'POST', `${route}/points/count`, { exact: true });
                        const sampleResponse = await qdrantBackend._request(
                            'POST', `${route}/points/scroll`, {
                                limit: 1,
                                with_payload: { include: ['embeddingSource'] },
                                with_vector: false,
                            });
                        const chunkCount = countResponse.result?.count;
                        if (!Number.isSafeInteger(chunkCount) || chunkCount < 0) {
                            throw new Error(`Invalid Qdrant count for ${collectionName}`);
                        }
                        const parts = collectionName.split(':');
                        const legacySource = parts.length === 2 ? parts[0]
                            : parts.length === 3 ? parts[1] : 'unknown';
                        const source = sampleResponse.result?.points?.[0]
                            ?.payload?.embeddingSource || legacySource;
                        allCollections.push({
                            id: collectionName, source, backend: 'qdrant',
                            chunkCount, modelCount: 1,
                        });
                    } catch (e) {
                        incomplete.add('qdrant');
                        console.warn(`[${pluginName}] Qdrant: Failed to read ${collectionName}:`, e.message);
                    }
                }
            }
        } catch (e) {
            incomplete.add('qdrant');
            console.warn(`[${pluginName}] Qdrant: Failed to scan collections:`, e.message);
        }

        // Scan Milvus
        try {
            if (milvusBackend.isConnected) {
                // The scan lists the shared physical collection as a whole.
                const items = await milvusBackend.listItems('vecthare_main', ALL_TENANTS, { limit: 1 });
                if (items.length > 0) {
                    const stats = await milvusBackend.getCollectionStats('vecthare_main', ALL_TENANTS);
                    allCollections.push({
                        id: 'vecthare_main',
                        source: 'milvus',
                        backend: 'milvus',
                        chunkCount: stats.chunkCount,
                        modelCount: 1
                    });
                }
            }
        } catch (e) {
            incomplete.add('milvus');
            console.warn(`[${pluginName}] Milvus: Failed to scan collections:`, e.message);
        }

    } catch (error) {
        for (const backend of ['vectra', 'lancedb', 'qdrant', 'milvus']) incomplete.add(backend);
        console.error(`[${pluginName}] Error scanning collections:`, error);
    }

    return { collections: allCollections, incomplete: [...incomplete] };
}

/**
 * VH-085: lists the requesting user's collections that live in v2 stores, from
 * the saved settings each store keeps per tenant. A record counts only if it
 * sits in the slot of THIS user's tenant for the collection id it names (the
 * tenant key contains the user's handle), so other users' collections are
 * never listed. Read-only. Each entry carries `storage` (layout, physical, dim).
 * @returns {Promise<{collections: object[], incomplete: string[]}>}
 */
async function listV2TenantCollections(req) {
    const collections = [];
    const incomplete = [];
    const decode = value => {
        let record = value;
        if (typeof record === 'string') { try { record = JSON.parse(record); } catch { return null; } }
        if (!record || typeof record !== 'object' || record.vecthareTombstone === 1) return null;
        if (typeof record.collectionId !== 'string' || record.snapshot?.schemaVersion !== 1) return null;
        return record;
    };
    const mine = (key, record) => {
        const tenant = tenantFor(req, record.collectionId);
        return tenant && configKeyForTenant(tenant.tenantKey) === key ? tenant : null;
    };
    try {
        if (qdrantBackend.baseUrl && await qdrantBackend.healthCheck()) {
            const listing = await qdrantBackend._request('GET', '/collections');
            const names = listing.result?.collections?.map(c => c.name);
            if (!Array.isArray(names)) throw new Error('Qdrant did not list its collections');
            for (const name of names.filter(n => MT_V2_NAME.test(n))) {
                try {
                    const info = await qdrantBackend._request('GET', `/collections/${name}`);
                    const metadata = info?.result?.config?.metadata || {};
                    const dim = Number(MT_V2_NAME.exec(name)[1]);
                    for (const [key, value] of Object.entries(metadata)) {
                        if (!key.startsWith('vecthare_initial_')) continue;
                        const record = decode(value);
                        const tenant = record && mine(key, record);
                        if (!tenant) continue;
                        const chunkCount = await qdrantBackend.countTenant(name, tenant.tenantKey);
                        collections.push({ id: record.collectionId, source: record.snapshot.embedding?.provider || 'transformers',
                            backend: 'qdrant', chunkCount, modelCount: 1, storage: { layout: 'mt-v2', physical: name, dim } });
                    }
                } catch (error) {
                    if (!incomplete.includes('qdrant')) incomplete.push('qdrant');
                    console.warn(`[${pluginName}] Qdrant: could not list the collections in ${name}:`, error.message);
                }
            }
        }
    } catch (error) {
        if (!incomplete.includes('qdrant')) incomplete.push('qdrant');
        console.warn(`[${pluginName}] Qdrant: v2 store listing failed:`, error.message);
    }
    try {
        if (milvusBackend.isConnected) {
            const names = await milvusBackend.listCollectionNames();
            for (const name of names.filter(n => MT_V2_NAME.test(n) || /^vh_[0-9a-f]{24}$/.test(n))) {
                try {
                    const described = await milvusBackend.describeProperties(name);
                    const layout = MT_V2_NAME.test(name) ? 'mt-v2' : 'per-collection';
                    for (const [key, value] of Object.entries(described.properties || {})) {
                        if (!key.startsWith('vecthare_initial_')) continue;
                        const record = decode(value);
                        const tenant = record && mine(key, record);
                        if (!tenant) continue;
                        const chunkCount = await milvusBackend.countV2(name, tenant.tenantKey);
                        collections.push({ id: record.collectionId, source: record.snapshot.embedding?.provider || 'transformers',
                            backend: 'milvus', chunkCount, modelCount: 1,
                            storage: { layout, physical: name, ...(Number.isSafeInteger(described.dim) ? { dim: described.dim } : {}) } });
                    }
                } catch (error) {
                    if (!incomplete.includes('milvus')) incomplete.push('milvus');
                    console.warn(`[${pluginName}] Milvus: could not list the collections in ${name}:`, error.message);
                }
            }
        }
    } catch (error) {
        if (!incomplete.includes('milvus')) incomplete.push('milvus');
        console.warn(`[${pluginName}] Milvus: v2 store listing failed:`, error.message);
    }
    return { collections, incomplete };
}

/**
 * Find all vectra index.json files
 */
async function findAllIndexes(dir, relativePath = '', onError = null) {
    const results = [];

    try {
        const entries = await fs.readdir(dir, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            const newRelativePath = relativePath ? path.join(relativePath, entry.name) : entry.name;

            if (entry.isDirectory()) {
                const subResults = await findAllIndexes(fullPath, newRelativePath, onError);
                results.push(...subResults);
            } else if (entry.name === 'index.json') {
                const pathParts = newRelativePath.split(path.sep);
                if (pathParts.length >= 3) {
                    results.push({
                        indexPath: fullPath,
                        collectionId: pathParts[1],
                        source: pathParts[0],
                        modelPath: pathParts.slice(2, -1).join(path.sep),
                        relativePath: newRelativePath
                    });
                }
            }
        }
    } catch (e) {
        // Directory may not exist or not be readable - log at debug level since this is recursive
        // (VH-008: an unreadable sub-folder, or an existing top folder, makes the listing incomplete).
        if (relativePath !== '' || e.code !== 'ENOENT') onError?.(e);
        if (relativePath === '') {
            console.warn(`[${pluginName}] Vectra: Failed to scan vectors directory:`, e.message);
        }
    }

    return results;
}

/**
 * Get chunk count from vectra index
 */
async function getChunkCountFromIndex(indexPath) {
    try {
        const modelDir = path.dirname(indexPath);
        const store = new vectra.LocalIndex(modelDir);

        if (!await store.isIndexCreated()) {
            return 0;
        }

        const items = await store.listItems();
        return items.length;
    } catch (e) {
        console.warn(`[${pluginName}] Vectra: Failed to get chunk count from ${indexPath}:`, e.message);
        return null; // VH-008: unknown, not zero
    }
}

export async function exit() {
    console.log(`[${pluginName}] Plugin shutting down...`);
}

export const info = {
    id: pluginName,
    name: 'Similharity',
    description: 'Unified vector database backend for VectHare - supports Vectra, LanceDB, and Qdrant',
    version: pluginVersion
};
