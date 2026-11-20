/**
 * ============================================================================
 * MILVUS BACKEND
 * ============================================================================
 * Server-side Milvus vector database operations.
 * Uses the official nodejs sdk.
 *
 * Multitenancy Strategy:
 * - ONE collection: "vecthare_main"
 * - Scalar fields for filtering: type, sourceId, etc.
 *
 * @author VectHare
 * @version 1.0.0
 * ============================================================================
 */

import { toSafeHashList, requireSafeHash, isSafeFilterString, isValidLiteral, quoteLiteral, badRequest } from './safe-values.js';

// Load the Milvus SDK only if this backend is selected.
let MilvusClient;
let DataType;

/**
 * Explicit opt-in for the one internal read that must see every logical
 * collection (the plugin's own collection scan). It is an identity-compared
 * object, so a request body can never produce it. Writes and deletes reject it.
 */
export const ALL_TENANTS = Object.freeze({ allTenants: true });

/**
 * Derive the tenant (type + sourceId) from a logical VectHare collection id,
 * the same way the extension's Milvus backend does:
 *   vecthare_{type}_{sourceId}   and the older   vh:{type}:{sourceId}
 * Strict: any other shape, unsafe character or over-long part returns null, and
 * callers must then refuse the request (never fall back to "everything").
 * @param {unknown} logicalId
 * @returns {{type: string, sourceId: string}|null}
 */
export function parseMilvusTenant(logicalId) {
    if (typeof logicalId !== 'string' || logicalId.length > 255) return null;
    // Same split as the extension: the type is the first token, the sourceId is
    // everything after it. Older collections are named after the chat file, so
    // the sourceId may contain spaces, '@', accents, etc.; it is escaped, never
    // interpolated raw (see quoteLiteral).
    const match = /^vecthare_([A-Za-z0-9.-]+)_(.+)$/s.exec(logicalId) ||
        /^vh:([A-Za-z0-9_.-]+):(.+)$/s.exec(logicalId);
    return match ? normalizeMilvusTenant({ type: match[1], sourceId: match[2] }) : null;
}

/**
 * Validate a tenant. Returns a clean {type, sourceId}, ALL_TENANTS (unchanged),
 * or null. 'unknown' is rejected: it is what un-attributed rows carry and would
 * span unrelated data.
 */
export function normalizeMilvusTenant(tenant) {
    if (tenant === ALL_TENANTS) return tenant;
    if (!tenant || typeof tenant !== 'object') return null;
    const { type, sourceId } = tenant;
    if (!isValidLiteral(type, 64) || !isValidLiteral(sourceId, 128)) return null;
    if (type === 'unknown' || sourceId === 'unknown') return null;
    return { type, sourceId };
}

/** VH-077: Milvus keeps chunk text in a VARCHAR(65535) field (characters). */
export const MILVUS_TEXT_MAX = 65535;
function assertMilvusTextFits(items) {
    for (const item of items) {
        const length = typeof item?.text === 'string' ? [...item.text].length : 0;
        if (length > MILVUS_TEXT_MAX) {
            throw badRequest(`This chunk is too long for Milvus (${length} characters; Milvus stores at most ${MILVUS_TEXT_MAX}). ` +
                'Shorten the message or use a smaller chunk size; nothing was written for it.');
        }
    }
}

/**
 * Milvus Backend Manager
 * Manages Milvus connection and operations
 */

/**
 * VH-034: the fields of a row (text, hash, vector) are not chunk notes; an
 * update must not copy them into the stored metadata (a vector there was sent
 * to the browser with every listing). A vector left in by older builds is dropped.
 */
function mergeRowMetadata(existingMetadata, updates) {
    const { text, hash, vector: _vector, ...notes } = updates || {};
    const merged = { ...(existingMetadata || {}), ...notes };
    // Where the stored metadata already mirrors the row's text/hash, keep the
    // mirror current; never add them where they were not.
    if (text !== undefined && Object.prototype.hasOwnProperty.call(merged, 'text')) merged.text = text;
    if (hash !== undefined && Object.prototype.hasOwnProperty.call(merged, 'hash')) merged.hash = hash;
    delete merged.vector;
    return merged;
}

class MilvusBackend {
    constructor() {
        this.client = null;
        // params must be stringified for some SDK versions
        this.baseParams = { 
            metric_type: 'COSINE', 
            index_type: 'IVF_FLAT', 
            params: JSON.stringify({ nlist: 1024 }) 
        };
        this.isConnected = false;
        this.config = {
            address: '127.0.0.1:19530',
            token: '', // API token or user:password
            ssl: false,
            dimensions: null,
        };
    }

    /**
     * Initialize Milvus connection
     * @param {object} config - Configuration { address, token, ssl, dimensions }
     */
    async initialize(config = {}) {
        // VH-001: replace (never merge) the configuration, and connect a candidate
        // client first; the working client is only swapped out on success.
        const next = {
            address: config.address || '127.0.0.1:19530',
            token: config.token || '',
            ssl: config.ssl === true,
            dimensions: config.dimensions ?? null,
        };

        console.log(`[Milvus] Initializing connection to ${next.address}...`);

        try {
            ({ MilvusClient, DataType } = await import('@zilliz/milvus2-sdk-node'));
            const candidate = new MilvusClient({
                address: next.address,
                token: next.token,
                ssl: next.ssl
            });

            // Test connection by listing collections
            await candidate.listCollections();
            const previous = this.client;
            const changed = !previous || next.address !== this.config.address || next.token !== this.config.token || next.ssl !== this.config.ssl;
            this.client = candidate;
            this.config = next;
            this.isConnected = true;
            this.v2Version = null; // a new connection may be a different server
            if (changed && previous && previous !== candidate) {
                // idle-release timers belong to the previous server
                for (const timer of this.v2Idle?.values() || []) clearTimeout(timer);
                this.v2Idle = new Map();
                Promise.resolve().then(() => previous.closeConnection?.()).catch(() => {});
            }
            console.log('[Milvus] Connection successful');
            
            // H1: an existing vecthare_main is only loaded here (no size check:
            // a configured or guessed size must never decide anything about stored
            // data). A missing one is created now only when the size is configured;
            // otherwise at the first insert, with the real vector size.
            const configured = Number(this.config.dimensions);
            await this.ensureCollection('vecthare_main', null);
            const { data: listed } = await this.client.listCollections();
            if (!listed.some(c => c.name === 'vecthare_main') && Number.isSafeInteger(configured) && configured > 0) {
                await this.ensureCollection('vecthare_main', configured);
            }
            // 9c: the new layouts need Milvus >= 2.2.9; say so once at start-up.
            this.assertV2Supported().catch(error => console.warn(error.message));
            
        } catch (error) {
            // The previous client (if any) is still in place and still connected.
            console.error('[Milvus] Connection failed:', error.message);
            throw error;
        }
    }

    /** 9a: collection names; errors propagate (never "no collections"). */
    async listCollectionNames() {
        if (!this.isConnected || !this.client) throw new Error('Milvus not initialized');
        const res = await this.client.listCollections();
        this._assertSuccess(res, 'List collections');
        if (!Array.isArray(res?.data)) throw new Error('Milvus returned no collection list');
        return res.data.map(c => c?.name).filter(name => typeof name === 'string');
    }

    /**
     * 9a: one describe of a collection - its properties as a plain object and
     * its vector size. Read-only.
     */
    async describeProperties(name) {
        if (!this.isConnected || !this.client) throw new Error('Milvus not initialized');
        const info = await this.client.describeCollection({ collection_name: name });
        if (info?.status && ((info.status.code !== undefined && Number(info.status.code) !== 0) ||
            (info.status.error_code !== undefined && !['Success', 0, '0'].includes(info.status.error_code)))) {
            throw new Error(`Milvus describe collection failed: ${info.status.reason || info.status.error_code || info.status.code}`);
        }
        if (!info || typeof info !== 'object' || (!info.schema && !info.properties)) {
            throw new Error('Milvus returned an invalid collection description');
        }
        const raw = info.properties ?? info.schema?.properties ?? [];
        const properties = {};
        if (Array.isArray(raw)) for (const p of raw) { if (p && typeof p.key === 'string') properties[p.key] = p.value; }
        else if (raw && typeof raw === 'object') Object.assign(properties, raw);
        const vectorField = info.schema?.fields?.find(f => f.name === 'vector');
        const dim = Number(vectorField?.dim || vectorField?.type_params?.find?.(p => p.key === 'dim')?.value ||
            vectorField?.params?.find?.(p => p.key === 'dim')?.value) || null;
        return { properties, dim };
    }

    /** Read a collection property without changing the shared collection. */
    async readInitialCollectionConfig(key) {
        if (!this.isConnected || !this.client) throw new Error('Milvus not initialized');
        // 9c fix: since H1 vecthare_main can be missing until the first insert
        // (no configured size). Describing it then failed, the read answered 503,
        // and a first Sync could never start. A missing store holds no settings.
        if (!await this._collectionExists()) return undefined;
        const info = await this.client.describeCollection({ collection_name: 'vecthare_main' });
        if (info?.status && ((info.status.code !== undefined && Number(info.status.code) !== 0) ||
            (info.status.error_code !== undefined && !['Success', 0, '0'].includes(info.status.error_code)))) {
            throw new Error(`Milvus describe collection failed: ${info.status.reason || info.status.error_code || info.status.code}`);
        }
        if (!info || typeof info !== 'object' || (!info.schema && !info.properties)) {
            throw new Error('Milvus returned an invalid collection description');
        }
        const properties = info.properties ?? info.schema?.properties ?? [];
        const value = Array.isArray(properties) ? properties.find(p => p.key === key)?.value
            : properties[key];
        // 9d: a moved-away collection's slot holds a tombstone; it reads as missing.
        return value !== undefined && String(value).includes('"vecthareTombstone":1') ? undefined : value;
    }

    /** 9d: empties one collection's saved-settings slot in vecthare_main (after a move). */
    async tombstoneLegacyConfig(key) {
        if (!this.isConnected || !this.client) throw new Error('Milvus not initialized');
        if (!await this._collectionExists()) return;
        const current = (await this.describeProperties('vecthare_main')).properties[key];
        if (current === undefined || String(current).includes('"vecthareTombstone":1')) return;
        this._assertSuccess(await this.client.alterCollectionProperties({ collection_name: 'vecthare_main',
            properties: { [key]: JSON.stringify({ vecthareTombstone: 1 }) } }), 'Clear saved settings');
    }

    /** Write-once provenance in properties of the shared Milvus collection. */
    async saveInitialCollectionConfig(key, value) {
        if (!this.isConnected || !this.client) throw new Error('Milvus not initialized');
        const name = 'vecthare_main';
        const getProperty = async () => {
            const info = await this.client.describeCollection({ collection_name: name });
            if (info?.status?.error_code && info.status.error_code !== 'Success') {
                throw new Error(`Milvus describe collection failed: ${info.status.reason || info.status.error_code}`);
            }
            const properties = info?.properties ?? info?.schema?.properties ?? [];
            if (Array.isArray(properties)) return properties.find(p => p.key === key)?.value;
            return properties[key];
        };
        const existing = await getProperty();
        if (existing !== undefined && !String(existing).includes('"vecthareTombstone":1')) {
            const original = JSON.parse(existing);
            if (original.collectionId !== value.collectionId || original.snapshot?.schemaVersion !== 1) {
                throw new Error('Existing Milvus initial configuration is invalid or belongs to another collection');
            }
            return false;
        }
        const encoded = JSON.stringify(value);
        const status = await this.client.alterCollectionProperties({
            collection_name: name, properties: { [key]: encoded },
        });
        if (status && ((status.code !== undefined && Number(status.code) !== 0) ||
            (status.error_code !== undefined && !['Success', 0, '0'].includes(status.error_code)))) {
            throw new Error(`Milvus refused collection metadata: ${status.reason || status.error_code}`);
        }
        if (await getProperty() !== encoded) {
            throw new Error('Milvus did not return the persisted initial configuration');
        }
        return true;
    }

    /**
     * Health check
     * @returns {Promise<boolean>}
     */
    async healthCheck() {
        try {
            if (!this.client) return false;
            await this.client.listCollections();
            return true;
        } catch (error) {
            console.error('[Milvus] Health check failed:', error.message);
            return false;
        }
    }

    /**
     * Ensure collection exists with proper schema
     * @param {string} collectionName - Collection name
     * @param {number} vectorSize - Vector dimension
     */
    async ensureCollection(collectionName, vectorSize = null) {
        if (!this.isConnected) throw new Error('Milvus not initialized');

        try {
            // Check if collection exists
            const { data: collections } = await this.client.listCollections();
            let exists = collections.find(c => c.name === collectionName);
            
            if (exists) {
                // H1: an existing collection is NEVER dropped here. It holds every
                // VectHare collection's vectors and their saved original settings;
                // a wrong or guessed size must stop the write, not erase them.
                let desc = null;
                try {
                    desc = await this.client.describeCollection({ collection_name: collectionName });
                } catch (e) {
                    console.error('[Milvus] Describe collection failed:', e.message);
                }
                const vectorField = desc?.schema?.fields?.find(f => f.name === 'vector');
                if (desc?.schema?.fields && !vectorField) {
                    const error = new Error(`[Milvus] Collection ${collectionName} has no 'vector' field; it was left untouched. ` +
                        'Fix or remove it in Milvus yourself if it is not VectHare data.');
                    error.status = 409;
                    throw error;
                }
                // Milvus returns dim as string or number depending on version.
                const existingDim = Number(vectorField?.dim || vectorField?.type_params?.find?.(p => p.key === 'dim')?.value ||
                    vectorField?.params?.find?.(p => p.key === 'dim')?.value);
                if (vectorSize != null && existingDim && existingDim !== Number(vectorSize)) {
                    const error = new Error(`[Milvus] ${collectionName} stores ${existingDim}-dimension vectors, but this ` +
                        `embedding model produces ${vectorSize}. Nothing was changed. Use a model with ${existingDim} ` +
                        'dimensions (or the original one) for this store.');
                    error.status = 409;
                    throw error;
                }

                if (exists) {
                    // Check if index exists
                    let indexExists = false;
                    try {
                        const indexInfo = await this.client.describeIndex({
                            collection_name: collectionName
                        });
                        if (indexInfo.status.error_code === 'Success') {
                            indexExists = true;
                            console.log(`[Milvus] Collection and index found for ${collectionName}`);
                        }
                    } catch (err) {
                        // Index missing
                    }

                    if (!indexExists) {
                        console.log(`[Milvus] Index missing for existing collection ${collectionName}, creating...`);
                        await this.client.createIndex({
                            collection_name: collectionName,
                            field_name: 'vector',
                            extra_params: this.baseParams
                        });
                    }

                    // Load collection
                    await this.client.loadCollection({
                        collection_name: collectionName
                    });
                    return;
                }
            }

            if (vectorSize == null) {
                // Nothing to verify or load yet; created at the first insert with
                // the real vector size.
                return;
            }
            console.log(`[Milvus] Creating collection ${collectionName}...`);

            // Define schema
            // Milvus 2.x requires specific fields definition
            const fields = [
                {
                    name: 'hash',
                    description: 'Primary key (hash of text)',
                    data_type: DataType.Int64,
                    is_primary_key: true,
                },
                {
                    name: 'vector',
                    description: 'Vector embedding',
                    data_type: DataType.FloatVector,
                    dim: vectorSize,
                },
                {
                    name: 'text',
                    description: 'Original text content',
                    data_type: DataType.VarChar,
                    max_length: 65535, // Max length for VarChar
                },
                // Metadata fields for filtering
                {
                    name: 'type', 
                    data_type: DataType.VarChar, 
                    max_length: 64,
                    description: 'Content type (chat, lorebook, etc)'
                },
                {
                    name: 'sourceId', 
                    data_type: DataType.VarChar, 
                    max_length: 128,
                    description: 'Source ID (chat_id, lorebook_id)'
                },
                {
                    name: 'embeddingSource', 
                    data_type: DataType.VarChar, 
                    max_length: 64
                },
                {
                    name: 'embeddingModel', 
                    data_type: DataType.VarChar, 
                    max_length: 128
                },
                {
                    name: 'timestamp',
                    data_type: DataType.Int64
                },
                {
                    name: 'metadata',
                    description: 'Full JSON metadata',
                    data_type: DataType.JSON 
                }
            ];

            await this.client.createCollection({
                collection_name: collectionName,
                fields: fields,
            });

            // Create index on vector field
            console.log('[Milvus] Creating index...');
            await this.client.createIndex({
                collection_name: collectionName,
                field_name: 'vector',
                extra_params: this.baseParams // Use default index params
            });

            // Load collection
            await this.client.loadCollection({
                collection_name: collectionName
            });

            console.log(`[Milvus] Created and loaded collection: ${collectionName} (dim=${vectorSize})`);

        } catch (error) {
            console.error(`[Milvus] Failed to ensure collection ${collectionName}:`, error.message);
            throw error;
        }
    }

    /**
     * Insert vectors into collection
     * @param {string} collectionName - Collection name
     * @param {Array} items - Items to insert
     * @param {object} tenantMetadata - Metadata for tenancy
     */
    async insertVectors(collectionName, items, tenantMetadata = {}) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        if (items.length === 0) return;
        assertMilvusTextFits(items);
        // Every row must carry the identity of the collection that owns it. A
        // row stamped 'unknown' could never be scoped, listed or deleted again.
        const tenant = normalizeMilvusTenant(tenantMetadata);
        if (!tenant || tenant === ALL_TENANTS) {
            throw badRequest('Milvus insert needs a valid collection identity (type and sourceId)');
        }

        const mainCollection = 'vecthare_main';

        // Check dimension from first item
        const vectorSize = items[0].vector?.length || 0;
        if (vectorSize === 0) throw new Error('[Milvus] Invalid vector dimension 0');

        // Ensure collection exists
        await this.ensureCollection(mainCollection, vectorSize);

        // VH-027: a plain insert does not replace a row with the same primary key
        // on a Milvus server, so a hash this collection already stores is skipped
        // (as insertV2 does) instead of being added again.
        const scope = this._buildFilterExpr(tenant);
        if (!scope) throw badRequest('Milvus insert needs a valid collection identity (type and sourceId)');
        this._assertSuccess(await this.client.loadCollection({ collection_name: mainCollection }), 'Load');
        const existing = new Set();
        const wanted = toSafeHashList(items.map(item => item.hash));
        for (let i = 0; i < wanted.length; i += 1000) {
            const res = await this.client.query({ collection_name: mainCollection, output_fields: ['hash'],
                filter: `${scope} && hash in [${wanted.slice(i, i + 1000).join(',')}]`, limit: 16384, consistency_level: 'Strong' });
            this._assertSuccess(res, 'Query');
            if (!Array.isArray(res?.data)) throw new Error('[Milvus] Invalid query response');
            for (const row of res.data) existing.add(String(Number(row.hash)));
        }
        const seen = new Set();
        items = items.filter(item => {
            const key = String(Number(item.hash));
            if (existing.has(key) || seen.has(key)) return false;
            seen.add(key);
            return true;
        });
        if (!items.length) return;

        // Prepare data matching schema
        const data = items.map(item => ({
            hash: item.hash,
            vector: item.vector,
            text: item.text,
            type: tenant.type,
            sourceId: tenant.sourceId,
            embeddingSource: tenantMetadata.embeddingSource || 'transformers',
            embeddingModel: tenantMetadata.embeddingModel || '',
            timestamp: item.metadata?.timestamp || Date.now(),
            metadata: {
                ...(item.metadata || {}),
                text: item.text, // Redundant but safe
                hash: item.hash
            }
        }));

        try {
            const res = await this.client.insert({
                collection_name: mainCollection,
                data: data
            });

            if (res.status.error_code !== 'Success') {
                console.error(`[Milvus] Insert returned error: ${res.status.error_code} - ${res.status.reason}`);
                // Use a simpler stringify for logging to avoid huge vector dumps
                const sample = { ...data[0], vector: '[vector data]' };
                console.error('[Milvus] Sample data causing error:', JSON.stringify(sample));
                throw new Error(res.status.reason || res.status.error_code);
            }
            
            // Explicitly flush to ensure data is visible immediately (crucial for Attu/Search visibility)
            await this.client.flush({ collection_names: [mainCollection] });
            
            console.log(`[Milvus] Inserted ${items.length} items into ${mainCollection}`);
        } catch (error) {
            console.error('[Milvus] Insert failed:', error.message);
            throw error;
        }
    }

    /**
     * Boolean expression that selects exactly one tenant's rows.
     * ALL_TENANTS yields '' (only reads may use it).
     */
    _scope(tenant) {
        const t = normalizeMilvusTenant(tenant);
        if (!t) throw badRequest('Milvus operation needs a valid collection identity (type and sourceId)');
        if (t === ALL_TENANTS) return '';
        return `type == ${quoteLiteral(t.type)} && sourceId == ${quoteLiteral(t.sourceId)}`;
    }

    /**
     * Hashes (from `hashes`) that ANOTHER collection also stores.
     *
     * The shared Milvus collection uses the text hash as its primary key and
     * Milvus removes rows by primary key, so a delete that is filtered to one
     * collection can still remove another collection's row that has the same
     * hash. Those hashes are returned here so the caller can refuse to delete
     * them rather than silently destroy another chat's vector.
     * @param {string[]} hashes Canonical integer strings
     * @param {{type:string, sourceId:string}} tenant
     * @returns {Promise<Set<string>>}
     */
    async _hashesHeldElsewhere(hashes, tenant) {
        const QUERY_LIMIT = 16384; // Milvus per-query row cap
        this._assertSuccess(await this.client.loadCollection({ collection_name: 'vecthare_main' }), 'Load');
        const res = await this.client.query({
            collection_name: 'vecthare_main',
            filter: `hash in [${hashes.join(',')}] && (type != ${quoteLiteral(tenant.type)} || sourceId != ${quoteLiteral(tenant.sourceId)})`,
            output_fields: ['hash'],
            limit: QUERY_LIMIT,
            consistency_level: 'Strong', // VH-013: decides what a delete may remove
        });
        this._assertSuccess(res, 'Shared-hash check');
        if (!Array.isArray(res?.data)) throw new Error('[Milvus] Invalid query response');
        if (res.data.length >= QUERY_LIMIT) {
            throw new Error('[Milvus] Too many other-collection rows share these hashes to verify a safe delete');
        }
        return new Set(res.data.map(row => String(Number(row.hash))));
    }

    /**
     * Delete vectors by hash, for ONE collection only.
     * @param {string} collectionName ignored (single physical collection)
     * @param {Array<number|string>} hashes validated as safe integers
     * @param {{type:string, sourceId:string}} tenant required
     */
    async deleteVectors(collectionName, hashes, tenant) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const t = normalizeMilvusTenant(tenant);
        if (!t || t === ALL_TENANTS) {
            throw badRequest('Milvus delete needs a valid collection identity (type and sourceId)');
        }
        const scope = this._scope(t);
        const list = toSafeHashList(hashes); // rejects the whole request on any bad entry
        if (list.length === 0) return;

        try {
            if (!await this._collectionExists()) return;
            const blocked = [];
            let deleted = 0;
            for (let i = 0; i < list.length; i += 1000) {
                const batch = list.slice(i, i + 1000);
                // Only hashes this collection really stores can reach another
                // collection's row; a hash it does not store matches nothing,
                // so a repeated delete of an already-removed hash stays harmless.
                const own = await this.client.query({
                    collection_name: 'vecthare_main',
                    filter: `${scope} && hash in [${batch.join(',')}]`,
                    output_fields: ['hash'],
                    limit: 16384,
                    consistency_level: 'Strong',
                });
                this._assertSuccess(own, 'Delete lookup');
                if (!Array.isArray(own?.data)) throw new Error('[Milvus] Invalid query response');
                const ownHashes = [...new Set(own.data.map(row => String(Number(row.hash))))];
                if (ownHashes.length === 0) continue;
                const shared = await this._hashesHeldElsewhere(ownHashes, t);
                const blockedNow = new Set(ownHashes.filter(h => shared.has(h)));
                blocked.push(...blockedNow);
                const safe = batch.filter(h => !blockedNow.has(h));
                if (!safe.length) continue;
                const res = await this.client.delete({
                    collection_name: 'vecthare_main',
                    filter: `${scope} && hash in [${safe.join(',')}]`,
                });
                this._assertSuccess(res, 'Delete');
                deleted += safe.length;
            }
            console.log(`[Milvus] Deleted ${deleted} item(s) from ${t.type}/${t.sourceId}`);
            if (blocked.length) {
                const error = new Error(
                    `[Milvus] ${blocked.length} chunk(s) were NOT deleted because another collection stores ` +
                    'the same text hash and the shared Milvus layout would remove both. ' +
                    'Per-collection Milvus storage (a later update) removes this limit.');
                error.status = 409;
                error.blockedHashes = blocked;
                throw error;
            }
        } catch (error) {
            console.error('[Milvus] Delete failed:', error.message);
            throw error; // never report a failed delete as success
        }
    }

    /** Absence is a legitimate empty result; any other failure must surface. */
    /**
     * Explicit existence of the shared physical collection. Errors propagate so
     * the caller can report 'unverified' instead of absence.
     * @returns {Promise<boolean>}
     */
    async physicalCollectionExists() {
        return await this._collectionExists();
    }

    async _collectionExists() {
        const res = await this.client.hasCollection({ collection_name: 'vecthare_main' });
        this._assertSuccess(res, 'Collection check');
        return res?.value === true;
    }

    /** Milvus can answer with a non-Success status instead of throwing. */
    _assertSuccess(res, action) {
        // Data responses carry `status`; status-only responses (load, drop) ARE the status.
        const status = res?.status ?? res;
        const code = status?.error_code;
        if (code !== undefined && code !== 'Success' && code !== 0) {
            throw new Error(`[Milvus] ${action} failed: ${status.reason || code}`);
        }
    }

    /**
     * Exact number of records matching the same filter listItems uses.
     * Independent of page size, so a truncated listing is detectable.
     */
    async countItems(filters) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        if (filters === ALL_TENANTS) {
            if (!await this._collectionExists()) return 0;
            const stats = await this.client.getCollectionStatistics({ collection_name: 'vecthare_main' });
            const rows = Number(stats?.data?.row_count);
            if (!Number.isSafeInteger(rows) || rows < 0) throw new Error('[Milvus] Invalid statistics response');
            return rows;
        }
        // A missing identity used to fall through to `hash > 0`, which counted
        // the whole shared store for every collection.
        const expr = this._buildFilterExpr(filters);
        if (!expr) throw badRequest('Milvus count needs a valid collection identity (type and sourceId)');
        if (!await this._collectionExists()) return 0;
        this._assertSuccess(await this.client.loadCollection({ collection_name: 'vecthare_main' }), 'Load');
        const res = await this._strongCount('vecthare_main', expr);
        this._assertSuccess(res, 'Count');
        const total = Number(res?.data);
        if (!Number.isSafeInteger(total) || total < 0) throw new Error('[Milvus] Invalid count response');
        return total;
    }

    /**
     * Build boolean expression for filtering
     */
    _buildFilterExpr(filters) {
        // ALL_TENANTS (internal scan) => no scope. Otherwise the collection
        // identity is mandatory and every value is validated before it is
        // placed in the expression.
        if (filters === ALL_TENANTS) return '';
        const scope = this._scope(filters);
        const conditions = [scope];
        if (filters.embeddingSource !== undefined && filters.embeddingSource !== '') {
            if (!isSafeFilterString(filters.embeddingSource, 64)) throw badRequest('Invalid embeddingSource filter');
            conditions.push(`embeddingSource == ${quoteLiteral(filters.embeddingSource)}`);
        }
        if (filters.timestampAfter !== undefined && filters.timestampAfter !== null && filters.timestampAfter !== '') {
            const ts = Number(filters.timestampAfter);
            if (!Number.isSafeInteger(ts)) throw badRequest('Invalid timestampAfter filter');
            conditions.push(`timestamp >= ${ts}`);
        }
        return conditions.join(' && ');
    }

    /**
     * Query/Search for similar vectors
     */
    async queryCollection(collectionName, queryVector, topK = 10, filters) {
        if (!this.isConnected) throw new Error('Milvus not initialized');

        const mainCollection = 'vecthare_main';
        const filterExpr = this._buildFilterExpr(filters); // requires a collection identity
        topK = Number.isSafeInteger(Number(topK)) && Number(topK) > 0 ? Math.min(Number(topK), 16384) : 10;

        try {
            if (!await this._collectionExists()) return [];
            // Need to verify collection is loaded
            this._assertSuccess(await this.client.loadCollection({ collection_name: mainCollection }), 'Load');

            const results = await this.client.search({
                collection_name: mainCollection,
                vector: queryVector,
                filter: filterExpr || undefined,
                limit: topK,
                output_fields: ['hash', 'text', 'metadata']
            });
            this._assertSuccess(results, 'Search');
            if (!Array.isArray(results?.results)) throw new Error('[Milvus] Invalid search response');

            return results.results.map(r => ({
                hash: Number(r.hash),
                text: r.text,
                score: r.score,
                metadata: r.metadata
            }));

        } catch (error) {
            console.error('[Milvus] Search failed:', error.message);
            throw error; // an empty result must mean "no matches", not "failed"
        }
    }

    /**
     * List items (Query by scalar)
     */
    async listItems(collectionName, filters, options = {}) {
        if (!this.isConnected) throw new Error('Milvus not initialized');

        const mainCollection = 'vecthare_main';
        // Scoped to one collection; only the internal scan passes ALL_TENANTS.
        const filterExpr = this._buildFilterExpr(filters) || 'hash > 0'; // Always need an expression for query

        try {
            if (!await this._collectionExists()) return [];
            this._assertSuccess(await this.client.loadCollection({ collection_name: mainCollection }), 'Load');

            const output_fields = ['hash', 'text', 'metadata'];
            if (options.includeVectors) output_fields.push('vector');

            // Milvus query doesn't support easy pagination like "scroll" in Qdrant
            // But we can use limit/offset
            const limit = Number.isSafeInteger(Number(options.limit)) && Number(options.limit) > 0
                ? Math.min(Number(options.limit), 16384) : 100;
            const offset = Number.isSafeInteger(Number(options.offset)) && Number(options.offset) >= 0
                ? Number(options.offset) : 0;

            const results = await this.client.query({
                collection_name: mainCollection,
                filter: filterExpr,
                output_fields: output_fields,
                limit: limit,
                offset: offset,
                consistency_level: 'Strong', // VH-013: lists feed moves and verification
            });
            this._assertSuccess(results, 'Query');
            if (!Array.isArray(results?.data)) throw new Error('[Milvus] Invalid query response');

            return results.data.map(r => ({
                hash: Number(r.hash),
                text: r.text,
                metadata: r.metadata,
                vector: options.includeVectors ? r.vector : undefined
            }));

        } catch (error) {
            console.error('[Milvus] List items failed:', error.message);
            throw error; // an empty list must mean "empty collection", not "failed"
        }
    }

    /**
     * Get single item
     */
    /**
     * P5: the rows of one collection for the given hashes (all physical copies),
     * read with a scoped `hash in [...]` query instead of a full listing.
     * @param {Array<number|string>} hashes Hashes to read
     * @param {object} filters Collection identity {type, sourceId}
     * @returns {Promise<Array<{hash:number, text:string, metadata:object}>>}
     */
    async getItemsByHashes(hashes, filters) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const mainCollection = 'vecthare_main';
        const scope = this._scope(filters);
        if (!scope) throw badRequest('Milvus read needs a valid collection identity (type and sourceId)');
        const list = toSafeHashList(hashes);
        if (!list.length) return [];
        if (!await this._collectionExists()) return [];
        this._assertSuccess(await this.client.loadCollection({ collection_name: mainCollection }), 'Load');
        const items = [];
        const LIMIT = 16384;
        for (let i = 0; i < list.length; i += 1000) {
            const batch = list.slice(i, i + 1000);
            const results = await this.client.query({
                collection_name: mainCollection,
                filter: `${scope} && hash in [${batch.join(', ')}]`,
                output_fields: ['hash', 'text', 'metadata'],
                limit: LIMIT,
                consistency_level: 'Strong',
            });
            this._assertSuccess(results, 'Get items by hash');
            if (!Array.isArray(results?.data)) throw new Error('[Milvus] Invalid query response');
            if (results.data.length >= LIMIT) throw new Error('[Milvus] Hash read may be incomplete');
            items.push(...results.data.map(r => ({ hash: Number(r.hash), text: r.text, metadata: r.metadata })));
        }
        return items;
    }

    async getItem(collectionName, hash, filters) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const mainCollection = 'vecthare_main';
        const safeHash = requireSafeHash(hash);
        const scope = this._scope(filters);
        if (!scope) throw badRequest('Milvus get needs a valid collection identity (type and sourceId)');

        try {
            if (!await this._collectionExists()) return null;
            this._assertSuccess(await this.client.loadCollection({ collection_name: mainCollection }), 'Load');
            const results = await this.client.query({
                collection_name: mainCollection,
                filter: `${scope} && hash == ${safeHash}`,
                output_fields: ['hash', 'text', 'vector', 'metadata'],
                limit: 1,
                consistency_level: 'Strong', // VH-013
            });
            this._assertSuccess(results, 'Get item');
            if (!Array.isArray(results?.data)) throw new Error('[Milvus] Invalid query response');

            if (results.data.length === 0) return null;
            const r = results.data[0];
            
            return {
                hash: Number(r.hash),
                text: r.text,
                vector: r.vector,
                metadata: r.metadata
            };

        } catch (error) {
            console.error('[Milvus] Get item failed:', error.message);
            throw error; // null must mean "not found", not "failed"
        }
    }

    /**
     * Update item (delete & insert)
     */
    async updateItem(collectionName, hash, updates, filters) {
        const existing = await this.getItem(collectionName, hash, filters);
        if (!existing) throw new Error(`Item ${hash} not found`);

        const newHash = updates.hash || hash;
        const newItem = {
            hash: newHash,
            text: updates.text || existing.text,
            vector: updates.vector || existing.vector,
            metadata: mergeRowMetadata(existing.metadata, updates)
        };

        if (String(newHash) !== String(hash)) {
            // Write the replacement first; remove the old hash only afterwards.
            await this.insertVectors(collectionName, [newItem], filters);
            await this.deleteVectors(collectionName, [hash], filters);
            return;
        }
        // Milvus insert does not replace an existing key: remove, add, and
        // restore the original if the add fails.
        await this.deleteVectors(collectionName, [hash], filters);
        try {
            await this.insertVectors(collectionName, [newItem], filters);
        } catch (error) {
            // VH-035: say so when the chunk could not be put back either.
            const restored = await this.insertVectors(collectionName, [existing], filters).then(() => true, () => false);
            if (!restored) {
                throw new Error(`[Milvus] The update failed and chunk ${hash} could not be put back: it is missing now. ` +
                    `Run Sync or Repair to store it again. (${error.message})`);
            }
            throw error;
        }
    }

    /**
     * Remove ONE collection's rows. The shared physical collection is never
     * dropped by this route: a request without a valid identity is refused.
     * Rows whose hash another collection also stores are not removed (see
     * deleteVectors) and make the purge fail loudly instead of reporting success.
     */
    async purgeCollection(collectionName, filters) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const t = normalizeMilvusTenant(filters);
        if (!t || t === ALL_TENANTS) {
            throw badRequest('Milvus purge needs a valid collection identity (type and sourceId)');
        }
        const scope = this._scope(t);

        try {
            if (!await this._collectionExists()) return { leftShared: 0 };
            this._assertSuccess(await this.client.loadCollection({ collection_name: 'vecthare_main' }), 'Load');
            // VH-011 (option B): in this older shared store the primary key is the
            // text hash, so a row whose text another collection also stores
            // cannot be deleted without deleting that collection's row too.
            // Decide up front which rows those are, delete every other row of
            // this collection, then confirm that exactly the shared ones remain.
            const { own, shared } = await this._ownAndSharedHashes(t);
            const unshared = own.filter(hash => !shared.has(hash));
            for (let i = 0; i < unshared.length; i += 1000) {
                const batch = unshared.slice(i, i + 1000);
                const res = await this.client.delete({
                    collection_name: 'vecthare_main',
                    filter: `${scope} && hash in [${batch.join(',')}]`,
                });
                this._assertSuccess(res, 'Delete');
            }
            const remaining = await this._allHashesIn('vecthare_main', scope);
            const unexpected = remaining.filter(hash => !shared.has(String(hash)));
            if (unexpected.length) {
                throw new Error(`[Milvus] Purge could not be confirmed: ${unexpected.length} row(s) remain for this collection`);
            }
            console.log(`[Milvus] Purged ${t.type}/${t.sourceId}` +
                (remaining.length ? ` (${remaining.length} row(s) shared with another collection left in place)` : ''));
            return { leftShared: remaining.length };
        } catch (error) {
            console.error('[Milvus] Purge failed:', error.message);
            throw error;
        }
    }

    /**
     * VH-011: this collection's hashes in the shared store, and which of them
     * another collection stores too.
     * @returns {Promise<{own: string[], shared: Set<string>}>}
     */
    async _ownAndSharedHashes(tenant) {
        const own = [...new Set((await this._allHashesIn('vecthare_main', this._scope(tenant))).map(h => String(Number(h))))];
        const shared = new Set();
        for (let i = 0; i < own.length; i += 1000) {
            for (const hash of await this._hashesHeldElsewhere(own.slice(i, i + 1000), tenant)) shared.add(String(hash));
        }
        return { own, shared };
    }

    /**
     * VH-011: rows of this collection whose text no other collection stores -
     * what a delete must have removed. 0 means deleted (shared rows that had to
     * stay do not count).
     */
    async countUnsharedItems(filters) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const t = normalizeMilvusTenant(filters);
        if (!t || t === ALL_TENANTS) throw badRequest('Milvus count needs a valid collection identity (type and sourceId)');
        if (!await this._collectionExists()) return { unshared: 0, shared: 0 };
        const { own, shared } = await this._ownAndSharedHashes(t);
        return { unshared: own.filter(hash => !shared.has(hash)).length, shared: own.filter(hash => shared.has(hash)).length };
    }

    /**
     * Get stats
     */
    async getCollectionStats(collectionName, filters) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const mainCollection = 'vecthare_main';

        try {
            // One collection's count, not the whole shared store's.
            let rowCount;
            if (filters === ALL_TENANTS) {
                const stats = await this.client.getCollectionStatistics({
                    collection_name: mainCollection
                });
                rowCount = stats.data.row_count;
            } else {
                this._scope(filters); // refuse a missing identity
                rowCount = await this.countItems(filters);
            }

            // Approximate token count if we can't query all efficiently
            // For now return basic stats
            return {
                chunkCount: Number(rowCount),
                backend: 'milvus'
            };

        } catch (error) {
            if (error.status === 400) throw error;
            return { chunkCount: 0, backend: 'milvus', error: error.message };
        }
    }

    // ========================================================================
    // 9c: Milvus v2 layouts
    //   mt-v2 (2b):          vecthare_mt_<dim>, tenantKey is the partition key
    //   per-collection (2a): vh_<24 hex>, one collection per VectHare collection
    // Both: auto-id primary key, `hash` INT64 with a scalar index, `tenantKey`
    // VarChar(512), text, metadata JSON, vector. Every operation is scoped by
    // the plugin-computed tenantKey. Needs Milvus >= 2.2.9 (partition keys).
    // ========================================================================

    /** Milvus server version check for the v2 layouts (cached once it passes). */
    async assertV2Supported() {
        if (!this.isConnected || !this.client) throw new Error('Milvus not initialized');
        if (this.v2Version?.ok) return;
        let version = null;
        try {
            const res = typeof this.client.getVersion === 'function' ? await this.client.getVersion() : null;
            version = res?.version ?? null;
        } catch { /* unknown below */ }
        const m = /(\d+)\.(\d+)(?:\.(\d+))?/.exec(String(version || ''));
        const v = m ? [Number(m[1]), Number(m[2]), Number(m[3] || 0)] : null;
        // Refused only when the server is KNOWN to be too old; an unknown version
        // is tried (Milvus itself rejects a partition key it does not support).
        const ok = !v || v[0] > 2 || (v[0] === 2 && (v[1] > 2 || (v[1] === 2 && v[2] >= 9)));
        this.v2Version = { ok, version };
        if (!v) console.warn('[Milvus] Server version unknown; the new storage layouts will be tried');
        if (!ok) {
            const error = new Error(`[Milvus] This storage layout needs Milvus 2.2.9 or newer (server reports ${version || 'an unknown version'}). Nothing was changed.`);
            error.status = 409;
            throw error;
        }
    }

    _v2Scope(tenantKey) {
        if (!isValidLiteral(tenantKey, 512)) throw badRequest('Invalid tenant scope');
        return `tenantKey == ${quoteLiteral(tenantKey)}`;
    }

    async _v2Exists(name) {
        const res = await this.client.hasCollection({ collection_name: name });
        this._assertSuccess(res, 'Collection check');
        return res?.value === true;
    }

    /** Per-collection (vh_*) stores are released after 10 minutes without use. */
    _v2Touch(name) {
        if (!/^vh_[0-9a-f]{24}$/.test(name)) return;
        this.v2Idle ??= new Map();
        clearTimeout(this.v2Idle.get(name));
        const timer = setTimeout(() => {
            this.v2Idle.delete(name);
            this.client?.releaseCollection({ collection_name: name }).catch(() => {});
        }, 10 * 60 * 1000);
        timer.unref?.();
        this.v2Idle.set(name, timer);
    }

    async _v2Load(name) {
        this._assertSuccess(await this.client.loadCollection({ collection_name: name }), 'Load');
        this._v2Touch(name);
    }

    /** Creates a v2 collection when missing; checks vector size and schema when present. */
    async ensureV2Collection(name, dim, { partitionKey }) {
        await this.assertV2Supported();
        if (await this._v2Exists(name)) {
            const desc = await this.client.describeCollection({ collection_name: name });
            const fields = desc?.schema?.fields || [];
            const vectorField = fields.find(f => f.name === 'vector');
            const existingDim = Number(vectorField?.dim || vectorField?.type_params?.find?.(p => p.key === 'dim')?.value ||
                vectorField?.params?.find?.(p => p.key === 'dim')?.value);
            if (!fields.some(f => f.name === 'tenantKey') || !vectorField) {
                const error = new Error(`[Milvus] ${name} is not a VectHare v2 collection; it was left untouched.`);
                error.status = 409;
                throw error;
            }
            if (existingDim && existingDim !== Number(dim)) {
                const error = new Error(`[Milvus] ${name} stores ${existingDim}-dimension vectors, but this model produces ${dim}. Nothing was changed.`);
                error.status = 409;
                throw error;
            }
            await this._v2Load(name);
            return;
        }
        const fields = [
            { name: 'id', data_type: DataType.Int64, is_primary_key: true, autoID: true },
            { name: 'hash', data_type: DataType.Int64 },
            { name: 'tenantKey', data_type: DataType.VarChar, max_length: 512, ...(partitionKey ? { is_partition_key: true } : {}) },
            { name: 'embeddingSource', data_type: DataType.VarChar, max_length: 64 },
            { name: 'timestamp', data_type: DataType.Int64 },
            { name: 'text', data_type: DataType.VarChar, max_length: 65535 },
            { name: 'metadata', data_type: DataType.JSON },
            { name: 'vector', data_type: DataType.FloatVector, dim: Number(dim) },
        ];
        this._assertSuccess(await this.client.createCollection({ collection_name: name, fields }), 'Create collection');
        this._assertSuccess(await this.client.createIndex({ collection_name: name, field_name: 'vector', extra_params: this.baseParams }), 'Create index');
        try {
            this._assertSuccess(await this.client.createIndex({ collection_name: name, field_name: 'hash', index_type: 'STL_SORT' }), 'Create hash index');
        } catch (error) {
            console.warn(`[Milvus] Scalar index on ${name}.hash not created (lookups still work, slower): ${error.message}`);
        }
        await this._v2Load(name);
        console.log(`[Milvus] Created v2 collection ${name} (dim=${dim}${partitionKey ? ', partition key tenantKey' : ''})`);
    }

    async _v2Query(name, filter, outputFields, limit = 16384, offset = 0) {
        const res = await this.client.query({ collection_name: name, filter, output_fields: outputFields,
            limit, offset, consistency_level: 'Strong' });
        this._assertSuccess(res, 'Query');
        if (!Array.isArray(res?.data)) throw new Error('[Milvus] Invalid query response');
        return res.data;
    }

    /** Hashes of this tenant that already exist, out of `hashes`. */
    async _v2ExistingHashes(name, tenantKey, hashes) {
        const found = new Set();
        for (let i = 0; i < hashes.length; i += 1000) {
            const batch = hashes.slice(i, i + 1000);
            const rows = await this._v2Query(name, `${this._v2Scope(tenantKey)} && hash in [${batch.join(',')}]`, ['hash']);
            for (const row of rows) found.add(String(Number(row.hash)));
        }
        return found;
    }

    /**
     * Insert into a v2 collection. Auto-id inserts are not idempotent, so
     * hashes this tenant already stores are skipped (the cross-tab lease and
     * the identity ledger rule out two writers at once).
     * @returns {Promise<number>} rows written
     */
    async insertV2(name, items, tenant, embedding = {}, { partitionKey, replacing = false }) {
        if (!items.length) return 0;
        assertMilvusTextFits(items);
        const dim = items[0].vector?.length || 0;
        if (!dim || items.some(item => item.vector?.length !== dim)) throw new Error('[Milvus] Invalid or mixed vector dimensions');
        await this.ensureV2Collection(name, dim, { partitionKey });
        const hashes = toSafeHashList(items.map(item => item.hash));
        // `replacing` (VH-035): the caller removes the old rows afterwards by id.
        const existing = replacing ? new Set() : await this._v2ExistingHashes(name, tenant.tenantKey, hashes);
        const seen = new Set();
        const rows = [];
        items.forEach((item, i) => {
            const hash = hashes[i];
            if (existing.has(hash) || seen.has(hash)) return;
            seen.add(hash);
            rows.push({
                hash: Number(hash), tenantKey: tenant.tenantKey,
                embeddingSource: String(embedding.embeddingSource || 'transformers').slice(0, 64),
                timestamp: Number.isSafeInteger(item.metadata?.timestamp) ? item.metadata.timestamp : Date.now(),
                text: item.text, vector: item.vector,
                metadata: { ...(item.metadata || {}), text: item.text, hash: Number(hash),
                    embeddingModel: embedding.embeddingModel || '', user: tenant.user },
            });
        });
        if (!rows.length) return 0;
        const res = await this.client.insert({ collection_name: name, data: rows });
        this._assertSuccess(res, 'Insert');
        await this.client.flush({ collection_names: [name] });
        this._v2Touch(name);
        return rows.length;
    }

    /** Deletes this tenant's rows for the given hashes (looked up by id first). */
    async deleteV2(name, hashes, tenantKey) {
        const list = toSafeHashList(hashes);
        if (!list.length || !await this._v2Exists(name)) return 0;
        await this._v2Load(name);
        let deleted = 0;
        for (let i = 0; i < list.length; i += 1000) {
            const rows = await this._v2Query(name, `${this._v2Scope(tenantKey)} && hash in [${list.slice(i, i + 1000).join(',')}]`, ['id']);
            const ids = rows.map(row => String(row.id)).filter(id => /^-?\d+$/.test(id));
            if (!ids.length) continue;
            const res = await this.client.delete({ collection_name: name, filter: `id in [${ids.join(',')}]` });
            this._assertSuccess(res, 'Delete');
            deleted += ids.length;
        }
        return deleted;
    }

    /**
     * VH-027: every hash in a scope, without the 16,384 offset+limit window.
     * Hash ranges are read whole; a range that fills a page is split in two.
     * The rows read must equal the scope's count, otherwise it throws.
     * @returns {Promise<number[]>} distinct hashes
     */
    async _allHashesIn(collectionName, scopeExpr) {
        const PAGE = 16384;
        const rowsIn = async (lo, hi) => {
            const res = await this.client.query({ collection_name: collectionName, output_fields: ['hash'],
                filter: `${scopeExpr} && hash > ${lo} && hash <= ${hi}`, limit: PAGE, consistency_level: 'Strong' });
            this._assertSuccess(res, 'Query');
            if (!Array.isArray(res?.data)) throw new Error('[Milvus] Invalid query response');
            return res.data;
        };
        const out = [];
        const walk = async (lo, hi) => {
            const rows = await rowsIn(lo, hi);
            if (rows.length < PAGE) { out.push(...rows.map(r => Number(r.hash))); return; }
            if (hi - lo <= 1n) throw new Error('[Milvus] Too many rows share one hash; the hash list cannot be read completely');
            const mid = lo + (hi - lo) / 2n;
            await walk(lo, mid);
            await walk(mid, hi);
        };
        // Bounds stay inside int64 (an out-of-range literal matches nothing). The
        // one value -2^63 is not covered; the count check below catches it.
        await walk(-(2n ** 63n), 2n ** 63n - 1n);
        const counted = await this._strongCount(collectionName, scopeExpr);
        this._assertSuccess(counted, 'Count');
        if (Number(counted?.data) !== out.length) {
            throw new Error(`[Milvus] Hash list incomplete (${out.length} of ${counted?.data} rows read)`);
        }
        return [...new Set(out)];
    }

    /** VH-027: all hashes of one legacy (vecthare_main) collection. */
    async listAllHashes(filters) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const expr = this._buildFilterExpr(filters);
        if (!expr) throw badRequest('Milvus listing needs a valid collection identity (type and sourceId)');
        if (!await this._collectionExists()) return [];
        this._assertSuccess(await this.client.loadCollection({ collection_name: 'vecthare_main' }), 'Load');
        return this._allHashesIn('vecthare_main', expr);
    }

    /** VH-027: all hashes of one tenant in a v2 collection. */
    async listAllHashesV2(name, tenantKey) {
        if (!await this._v2Exists(name)) return [];
        await this._v2Load(name);
        return this._allHashesIn(name, this._v2Scope(tenantKey));
    }

    /**
     * VH-013: counts with Strong consistency, like the lists and deletes they are
     * compared with. The SDK's client.count() drops consistency_level, so the
     * count is made with query() and count(*), the same request count() sends.
     * @returns {Promise<{status: object, data: number}>}
     */
    async _strongCount(collectionName, expr) {
        const res = await this.client.query({ collection_name: collectionName, expr, filter: expr,
            output_fields: ['count(*)'], consistency_level: 'Strong' });
        const row = Array.isArray(res?.data) ? res.data[0] : null;
        return { status: res?.status, data: row ? Number(row['count(*)']) : NaN };
    }

    async countV2(name, tenantKey) {
        if (!await this._v2Exists(name)) return 0;
        await this._v2Load(name);
        const expr = this._v2Scope(tenantKey);
        const res = await this._strongCount(name, expr);
        this._assertSuccess(res, 'Count');
        const total = Number(res?.data);
        if (!Number.isSafeInteger(total) || total < 0) throw new Error('[Milvus] Invalid count response');
        return total;
    }

    async listV2(name, tenantKey, options = {}) {
        if (!await this._v2Exists(name)) return [];
        await this._v2Load(name);
        const limit = Number.isSafeInteger(Number(options.limit)) && Number(options.limit) > 0 ? Math.min(Number(options.limit), 16384) : 100;
        const offset = Number.isSafeInteger(Number(options.offset)) && Number(options.offset) >= 0 ? Number(options.offset) : 0;
        const fields = ['hash', 'text', 'metadata', ...(options.includeVectors ? ['vector'] : [])];
        const rows = await this._v2Query(name, this._v2Scope(tenantKey), fields, limit, offset);
        return rows.map(r => ({ hash: Number(r.hash), text: r.text, metadata: r.metadata,
            vector: options.includeVectors ? r.vector : undefined }));
    }

    async getByHashesV2(name, tenantKey, hashes, { withVector = false } = {}) {
        const list = toSafeHashList(hashes);
        if (!list.length || !await this._v2Exists(name)) return [];
        await this._v2Load(name);
        const out = [];
        for (let i = 0; i < list.length; i += 1000) {
            const rows = await this._v2Query(name, `${this._v2Scope(tenantKey)} && hash in [${list.slice(i, i + 1000).join(',')}]`,
                ['hash', 'text', 'metadata', ...(withVector ? ['vector'] : [])]);
            out.push(...rows.map(r => ({ hash: Number(r.hash), text: r.text, metadata: r.metadata, ...(withVector ? { vector: r.vector } : {}) })));
        }
        return out;
    }

    async queryV2(name, tenantKey, vector, topK = 10) {
        if (!await this._v2Exists(name)) return [];
        await this._v2Load(name);
        topK = Number.isSafeInteger(Number(topK)) && Number(topK) > 0 ? Math.min(Number(topK), 16384) : 10;
        const results = await this.client.search({ collection_name: name, vector, filter: this._v2Scope(tenantKey),
            limit: topK, output_fields: ['hash', 'text', 'metadata'] });
        this._assertSuccess(results, 'Search');
        if (!Array.isArray(results?.results)) throw new Error('[Milvus] Invalid search response');
        return results.results.map(r => ({ hash: Number(r.hash), text: r.text, score: r.score, metadata: r.metadata }));
    }

    async updateV2(name, hash, updates, tenant, embedding, options) {
        const [existing] = await this.getByHashesV2(name, tenant.tenantKey, [hash], { withVector: true });
        if (!existing) throw new Error(`Item ${hash} not found`);
        const newHash = updates.hash || hash;
        const item = { hash: newHash, text: updates.text || existing.text, vector: updates.vector || existing.vector,
            metadata: mergeRowMetadata(existing.metadata, updates) };
        if (String(newHash) !== String(hash)) {
            await this.insertV2(name, [item], tenant, embedding, options);
            await this.deleteV2(name, [hash], tenant.tenantKey);
            return;
        }
        // VH-035: rows have their own ids, so the new row is written first and
        // the old one removed afterwards by id. A failed write leaves the old
        // row untouched; a failed removal leaves both, and the next attempt
        // (or Repair's duplicate check) settles it. The chunk is never missing.
        await this._v2Load(name);
        const oldRows = await this._v2Query(name,
            `${this._v2Scope(tenant.tenantKey)} && hash in [${toSafeHashList([hash]).join(',')}]`, ['id']);
        const oldIds = oldRows.map(row => String(row.id)).filter(id => /^-?\d+$/.test(id));
        if (!oldIds.length) throw new Error(`Item ${hash} not found`);
        const written = await this.insertV2(name, [item], tenant, embedding, { ...options, replacing: true });
        if (written !== 1) throw new Error('[Milvus] The updated chunk was not written; the old one is unchanged');
        const res = await this.client.delete({ collection_name: name, filter: `id in [${oldIds.join(',')}]` });
        this._assertSuccess(res, 'Delete');
    }

    /**
     * Removes one collection. mt-v2: this tenant's rows, then a check that none
     * remain, then a tombstone in its saved-settings slot (the shared collection
     * stays). per-collection: drops its OWN collection (and its settings with it).
     */
    async purgeV2(name, tenantKey, { drop, configKey }) {
        if (!await this._v2Exists(name)) return;
        if (drop) {
            if (!/^vh_[0-9a-f]{24}$/.test(name)) throw new Error('[Milvus] Only a per-collection store may be dropped');
            this._assertSuccess(await this.client.dropCollection({ collection_name: name }), 'Drop');
            clearTimeout(this.v2Idle?.get(name));
            this.v2Idle?.delete(name);
            return;
        }
        await this._v2Load(name);
        for (let guard = 0; guard < 100000; guard++) {
            const rows = await this._v2Query(name, this._v2Scope(tenantKey), ['id'], 1000);
            if (!rows.length) break;
            const res = await this.client.delete({ collection_name: name, filter: `id in [${rows.map(r => String(r.id)).join(',')}]` });
            this._assertSuccess(res, 'Purge');
        }
        if (await this.countV2(name, tenantKey) !== 0) throw new Error('[Milvus] Purge could not be confirmed: rows remain');
        if (configKey) {
            const current = (await this.describeProperties(name)).properties[configKey];
            if (current !== undefined && !String(current).includes('"vecthareTombstone":1')) {
                const status = await this.client.alterCollectionProperties({ collection_name: name,
                    properties: { [configKey]: JSON.stringify({ vecthareTombstone: 1 }) } });
                this._assertSuccess(status, 'Clear saved settings');
            }
        }
    }

    async readConfigV2(name, key) {
        if (!await this._v2Exists(name)) return undefined;
        const value = (await this.describeProperties(name)).properties[key];
        if (value === undefined) return undefined;
        return String(value).includes('"vecthareTombstone":1') ? undefined : value;
    }

    /** Write-once (a tombstone may be replaced). @returns {Promise<boolean>} written */
    async saveConfigV2(name, key, value) {
        const current = (await this.describeProperties(name)).properties[key];
        if (current !== undefined && !String(current).includes('"vecthareTombstone":1')) {
            const original = JSON.parse(current);
            if (original.collectionId !== value.collectionId || original.snapshot?.schemaVersion !== 1) {
                throw new Error('Existing Milvus initial configuration is invalid or belongs to another collection');
            }
            return false;
        }
        const encoded = JSON.stringify(value);
        this._assertSuccess(await this.client.alterCollectionProperties({ collection_name: name, properties: { [key]: encoded } }),
            'Save collection settings');
        if ((await this.describeProperties(name)).properties[key] !== encoded) {
            throw new Error('Milvus did not return the persisted initial configuration');
        }
        return true;
    }
}

const milvusBackend = new MilvusBackend();
export default milvusBackend;
