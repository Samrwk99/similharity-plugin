/**
 * ============================================================================
 * QDRANT BACKEND (DIRECT REST API)
 * ============================================================================
 * Server-side Qdrant vector database operations using direct REST API calls.
 * Uses ONE main collection with payload filters for different data types.
 *
 * Why Direct REST instead of SDK?
 * - More transparent error handling
 * - Works better with various Qdrant setups
 * - Explicit control over headers and requests
 * - Better debugging capabilities
 *
 * Note: Qdrant Cloud may have CORS issues when accessed from browser.
 * This backend runs server-side so CORS is not an issue here.
 *
 * Multitenancy Strategy:
 * - ONE collection: "vecthare_main"
 * - Payload fields: type, sourceId, timestamp, etc.
 * - Filters for isolation: {type: "chat", sourceId: "chat_001"}
 *
 * @author VectHare
 * @version 3.0.0
 * ============================================================================
 */

import { uuidv5, MT_V2_NAME } from './tenancy.js';

/**
 * Qdrant Backend Manager
 * Manages Qdrant REST API connection and operations
 */
const KNOWN_EMBEDDING_SOURCES = new Set([
    'transformers', 'openai', 'cohere', 'ollama', 'llamacpp',
    'vllm', 'koboldcpp', 'webllm', 'bananabread', 'openrouter',
    'electronhub', 'mistral', 'nomicai', 'palm', 'vertexai',
    'togetherai', 'extras',
]);

/** H2: physical Qdrant/Milvus collections shared by many VectHare collections. */
export function isReservedSharedCollection(name) {
    return typeof name === 'string' && /^(vecthare_main|vecthare_multitenancy|vecthare_mt_\d+)$/.test(name);
}

/** A logical collection id as legacy multitenancy stores it in `content_type`. */
export function isLegacyContentType(value) {
    return typeof value === 'string' && value.length <= 255 && /^vecthare_[^\u0000-\u001f]+$/.test(value);
}

/**
 * The old extension sends its legacy-multitenancy scope as
 * `filter: { must: [{ key: 'content_type', match: { value } }] }` (singular
 * `filter`, which the plugin never read). Returns that value, or null.
 */
export function legacyContentTypeFromFilter(filter) {
    const must = Array.isArray(filter?.must) ? filter.must : [];
    const hits = must.filter(c => c?.key === 'content_type');
    if (hits.length !== 1) return null;
    const value = hits[0]?.match?.value;
    return isLegacyContentType(value) ? value : null;
}

/** VH-015: first Qdrant version that keeps collection metadata (saved settings). */
export const QDRANT_MIN_VERSION = '1.15';


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

class QdrantBackend {
    /**
     * 9c: extra scope conditions. `tenantKey` (v2 layouts, set by the plugin's
     * route guard only) and `content_type` (legacy multitenancy, when the caller
     * asked for it explicitly). Empty for every other request, so legacy request
     * bodies are unchanged.
     */
    _extraScope(filters = {}) {
        const must = [];
        if (typeof filters?.tenantKey === 'string' && filters.tenantKey) {
            must.push({ key: 'tenantKey', match: { value: filters.tenantKey } });
        }
        if (typeof filters?.content_type === 'string' && filters.content_type) {
            must.push({ key: 'content_type', match: { value: filters.content_type } });
        }
        return must;
    }

    _isNotFound(error) {
        // Other 404s (unsupported endpoint, proxy route, etc.) are real errors.
        return error?.status === 404 &&
            /(?:collection.{0,120}(?:does(?:\s+not|n't)\s+exist|not\s+found)|not\s+found:\s*collection)/i.test(String(error.message));
    }

    constructor() {
        this.baseUrl = null;
        this.apiKey = null;
        this.config = {
            host: '127.0.0.1',
            port: 6333,
            url: null,
            apiKey: null,
        };
    }

    /**
     * Parse collection name to handle both old (source:id) and new (backend:source:id) formats
     * @param {string} collectionName - Collection name in format "qdrant:source:id" or "source:id"
     * @returns {string} - Actual collection name (source:id)
     */
    _parseCollectionName(collectionName) {
        if (!collectionName) return collectionName;
        if (collectionName.startsWith('qdrant:')) {
            return collectionName.slice('qdrant:'.length);
        }
        const separator = collectionName.indexOf(':');
        if (separator !== -1 && KNOWN_EMBEDDING_SOURCES.has(collectionName.slice(0, separator))) {
            return collectionName.slice(separator + 1);
        }
        return collectionName;
    }

    /**
     * Build headers for Qdrant API requests
     */
    _getHeaders() {
        const headers = {
            'Content-Type': 'application/json',
        };
        if (this.apiKey) {
            headers['api-key'] = this.apiKey;
        }
        return headers;
    }

    /**
     * Make a request to Qdrant API with retry logic
     * @param {string} method - HTTP method
     * @param {string} endpoint - API endpoint
     * @param {object|null} body - Request body
     * @param {number} maxRetries - Maximum retry attempts for transient failures
     */
    async _request(method, endpoint, body = null, maxRetries = 3, timeoutMs = 60000) {
        const url = `${this.baseUrl}${endpoint}`;

        let lastError;
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
            const options = {
                method,
                headers: this._getHeaders(),
                signal: controller.signal,
            };
            if (body) {
                options.body = JSON.stringify(body);
            }
            try {
                const response = await fetch(url, options);

                if (!response.ok) {
                    const errorText = await response.text();
                    let errorMessage;
                    try {
                        const errorJson = JSON.parse(errorText);
                        errorMessage = errorJson.status?.error || errorJson.message || errorText;
                    } catch {
                        errorMessage = errorText;
                    }

                    // Check if error is retryable (5xx errors, 429 rate limit)
                    const isRetryable = response.status >= 500 || response.status === 429;
                    if (isRetryable && attempt < maxRetries) {
                        const delay = Math.min(1000 * Math.pow(2, attempt), 10000); // Exponential backoff, max 10s
                        console.warn(`[Qdrant] ${method} ${endpoint} failed (${response.status}), retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})`);
                        await new Promise(resolve => setTimeout(resolve, delay));
                        continue;
                    }

                    const error = new Error(`Qdrant ${method} ${endpoint} failed: ${response.status} ${errorMessage}`);
                    error.status = response.status;
                    throw error;
                }

                // Some endpoints return empty response
                const text = await response.text();
                return text ? JSON.parse(text) : null;

            } catch (error) {
                lastError = error;

                // Network errors (ECONNREFUSED, timeout, etc.) are retryable
                const isNetworkError = error.name === 'TypeError' ||
                    error.name === 'AbortError' ||
                    error.message?.includes('fetch failed') ||
                    error.message?.includes('ECONNREFUSED') ||
                    error.message?.includes('ETIMEDOUT');

                if (isNetworkError && attempt < maxRetries) {
                    const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
                    console.warn(`[Qdrant] Network error, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries}): ${error.message}${error.cause ? ` (Cause: ${error.cause})` : ''}`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }

                throw error;
            } finally {
                // Ensure timeout is cleared on success or error thrown
                clearTimeout(timeoutId);
            }
        }

        throw lastError;
    }

    /**
     * Initialize Qdrant connection
     * @param {object} config - Configuration { host, port, url, apiKey }
     */
    async initialize(config = {}) {
        // VH-001: the new configuration REPLACES the old one (an omitted key is
        // cleared, never inherited from whoever initialised before), and it is
        // tested on its own before it is swapped in: a failed init leaves the
        // working connection untouched.
        const next = {
            host: config.host || '127.0.0.1',
            port: config.port || 6333,
            url: config.url || null,
            apiKey: config.apiKey || null,
        };
        const apiKey = next.apiKey || null;
        const baseUrl = next.url ? String(next.url).replace(/\/+$/, '') : `http://${next.host}:${next.port}`;

        console.log('[Qdrant] Initializing with URL:', baseUrl);
        console.log('[Qdrant] API Key:', apiKey ? '(set)' : '(not set)');

        // Test connection with the candidate settings only
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000);
        try {
            const response = await fetch(`${baseUrl}/collections`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'api-key': apiKey } : {}) },
                signal: controller.signal,
            });
            if (!response.ok) {
                const text = await response.text().catch(() => '');
                let message = text;
                try { const json = JSON.parse(text); message = json.status?.error || json.message || text; } catch { /* text */ }
                const error = new Error(`Qdrant GET /collections failed: ${response.status} ${message}`);
                error.status = response.status;
                throw error;
            }
            console.log('[Qdrant] Connection successful');
        } catch (error) {
            const reason = error.name === 'AbortError' ? 'timed out after 15 s' : error.message;
            console.error('[Qdrant] Connection failed:', reason);
            throw error.name === 'AbortError' ? new Error(`Qdrant connection to ${baseUrl} timed out`) : error;
        } finally {
            clearTimeout(timeoutId);
        }

        const changed = baseUrl !== this.baseUrl || apiKey !== this.apiKey;
        this.config = next;
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        if (changed) this.v2Ensured = new Set(); // memo of the previous server

        // VH-015: saved settings live in collection metadata, which Qdrant keeps
        // only from 1.15 on (older versions accept the PATCH and drop it).
        this.serverVersion = null;
        try {
            const root = await this._request('GET', '/', null, 0);
            if (typeof root?.version === 'string') this.serverVersion = root.version;
        } catch { /* unknown version: the read-back check below still protects writes */ }
        if (this.serverVersion && !this.supportsCollectionMetadata()) {
            console.warn(`[Qdrant] Server ${this.serverVersion} is older than ${QDRANT_MIN_VERSION}: VectHare can read but not write collections`);
        }

        // Ensure indexes exist on any existing vecthare_main collection
        await this.ensurePayloadIndexes('vecthare_main');
        const collections = await this.getCollections();
        for (const collectionName of collections || []) {
            await this.ensurePayloadIndexes(collectionName);
        }
    }

    /** Read the original configuration without creating or updating a collection. */
    async readInitialCollectionConfig(physicalName, key) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        const name = encodeURIComponent(this._parseCollectionName(physicalName));
        let info;
        try {
            info = await this._request('GET', `/collections/${name}`);
        } catch (error) {
            // A genuinely absent collection is normal before its first insert.
            // Do not mask auth, timeout, or other server failures as absence.
            if (this._isNotFound(error)) return undefined;
            throw error;
        }
        if (!info?.result?.config || typeof info.result.config !== 'object') {
            throw new Error('Qdrant returned an invalid collection description');
        }
        const metadata = info.result.config.metadata;
        if (metadata != null && (typeof metadata !== 'object' || Array.isArray(metadata))) {
            throw new Error('Qdrant returned invalid collection metadata');
        }
        const value = metadata?.[key];
        // 9c: a purged v2 tenant leaves a tombstone; it reads as "missing".
        return value && value.vecthareTombstone === 1 ? undefined : value;
    }

    /** VH-015: true when the server keeps collection metadata (Qdrant 1.15+), null when unknown. */
    supportsCollectionMetadata() {
        if (!this.serverVersion) return null;
        const [a, b] = this.serverVersion.split('.').map(n => parseInt(n, 10));
        const [ma, mb] = QDRANT_MIN_VERSION.split('.').map(Number);
        return a > ma || (a === ma && b >= mb);
    }

    /** VH-015: the refusal used before anything is written on a too-old server. */
    tooOldMessage() {
        return `VectHare needs Qdrant ${QDRANT_MIN_VERSION} or newer to store collections; this Qdrant server runs ${this.serverVersion}. ` +
            'Update Qdrant; nothing was written.';
    }

    /** Write-once provenance stored in Qdrant's native collection metadata (Qdrant 1.15+). */
    async saveInitialCollectionConfig(physicalName, key, value) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        if (this.supportsCollectionMetadata() === false) throw new Error(this.tooOldMessage());
        const name = encodeURIComponent(this._parseCollectionName(physicalName));
        const endpoint = `/collections/${name}`;
        const info = await this._request('GET', endpoint);
        const metadata = info?.result?.config?.metadata || {};
        if (metadata[key] !== undefined && !(metadata[key] && metadata[key].vecthareTombstone === 1)) {
            if (metadata[key]?.collectionId !== value.collectionId ||
                metadata[key]?.snapshot?.schemaVersion !== 1) {
                throw new Error('Existing Qdrant initial configuration is invalid or belongs to another collection');
            }
            return false;
        }
        await this._request('PATCH', endpoint, { metadata: { [key]: value } });
        const written = (await this._request('GET', endpoint))?.result?.config?.metadata?.[key];
        if (JSON.stringify(written) !== JSON.stringify(value)) {
            throw new Error(`Qdrant did not keep the saved collection settings (collection metadata needs Qdrant ${QDRANT_MIN_VERSION} or newer)`);
        }
        return true;
    }

    /**
     * Health check
     * @returns {Promise<boolean>}
     */
    async healthCheck() {
        try {
            if (!this.baseUrl) return false;
            await this._request('GET', '/collections');
            return true;
        } catch (error) {
            console.error('[Qdrant] Health check failed:', error.message);
            return false;
        }
    }

    /**
     * Ensure collection exists with proper schema and payload indexes
     * @param {string} collectionName - Collection name
     * @param {number} vectorSize - Vector dimension (e.g., 768)
     */
    async ensureCollection(collectionName, vectorSize = 768) {
        try {
            // Check if collection exists
            const collections = await this._request('GET', '/collections');
            collectionName = this._parseCollectionName(collectionName);
            const exists = collections.result?.collections?.some(c => c.name === collectionName);
            if (!exists) {
                // Create collection
                await this._request('PUT', `/collections/${collectionName}`, {
                    vectors: {
                        size: vectorSize,
                        distance: 'Cosine',
                    },
                });
                console.log(`[Qdrant] Created collection: ${collectionName} (dim=${vectorSize})`);

                // Create payload indexes for filterable fields
                await this.createPayloadIndexes(collectionName);
            }
        } catch (error) {
            console.error(`[Qdrant] Failed to ensure collection ${collectionName}:`, error.message);
            throw error;
        }
    }

    /**
     * Create payload indexes for filterable fields
     * @param {string} collectionName - Collection name
     */
    async createPayloadIndexes(collectionName) {
        collectionName = this._parseCollectionName(collectionName);
        // Fields that need indexes for filtering
        // Tenant fields use { type: 'keyword', is_tenant: true } for optimized multitenancy
        // Regular fields use simple type string
        const indexConfigs = [
            // TENANT FIELDS (is_tenant: true for optimized multitenancy)
            { field: 'type', schema: { type: 'keyword', is_tenant: true } },
            { field: 'sourceId', schema: { type: 'keyword', is_tenant: true } },
            // REGULAR KEYWORD FIELDS
            { field: 'content_type', schema: { type: 'keyword' } },
            { field: 'embeddingSource', schema: 'keyword' },
            { field: 'embeddingModel', schema: 'keyword' },
            { field: 'characterName', schema: 'keyword' },
            { field: 'chatId', schema: 'keyword' },
            { field: 'keywords', schema: 'keyword' }, // legacy string keyword arrays
            { field: 'keywords[].text', schema: { type: 'text', tokenizer: 'word', lowercase: true } }, // keyword objects
            // TEXT FIELDS (for full-text search)
            { field: 'text', schema: 'text' },
            // INTEGER FIELDS
            { field: 'hash', schema: 'integer' },
            { field: 'timestamp', schema: 'integer' },
            { field: 'importance', schema: 'integer' },
        ];

        const info = await this._request('GET', `/collections/${collectionName}`);
        const existingIndexes = info.result?.payload_schema || {};
        for (const { field, schema } of indexConfigs) {
            const existing = existingIndexes[field];
            const type = typeof schema === 'string' ? schema : schema.type;
            if (existing?.data_type === type &&
                (typeof schema === 'string' || Object.entries(schema).every(
                    ([key, value]) => key === 'type' || existing.params?.[key] === value))) {
                continue;
            }
            try {
                await this._request('PUT', `/collections/${collectionName}/index`, {
                    field_name: field,
                    field_schema: schema,
                });
                const schemaType = typeof schema === 'object' ? `${schema.type}${schema.is_tenant ? ' (tenant)' : ''}` : schema;
                console.log(`[Qdrant] Created index for ${field} (${schemaType})`);
            } catch (error) {
                // Index might already exist, that's fine
                // VH-069: decided on the HTTP status, not on "409" anywhere in the text.
                if (error.status !== 409 && !/already exists/i.test(error.message || '')) {
                    console.warn(`[Qdrant] Failed to create index for ${field}:`, error.message);
                }
            }
        }
    }

    /**
     * Ensure payload indexes exist on an existing collection
     * @param {string} collectionName - Collection name
     */
    async ensurePayloadIndexes(collectionName) {
        try {
            await this.createPayloadIndexes(collectionName);
        } catch (error) {
            if (this._isNotFound(error)) return;
            console.error(`[Qdrant] Failed to ensure indexes for ${collectionName}:`, error.message);
        }
    }

    /**
     * Insert vector items into collection (MULTITENANCY)
     * @param {string} collectionName - Collection name (always "vecthare_main")
     * @param {Array} items - Items with {hash, text, vector, metadata}
     * @param {object} tenantMetadata - Tenant info {type, sourceId, embeddingSource}
     * @returns {Promise<void>}
     */
    async insertVectors(collectionName, items, tenantMetadata = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        if (items.length === 0) return;

        // MULTITENANCY: Always use vecthare_main collection
        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        // Validate all items have vectors
        const missingVectors = items.filter(item => !item.vector || !Array.isArray(item.vector));
        if (missingVectors.length > 0) {
            throw new Error(`[Qdrant] ${missingVectors.length} items are missing vectors. Hashes: ${missingVectors.slice(0, 5).map(i => i.hash).join(', ')}${missingVectors.length > 5 ? '...' : ''}`);
        }

        // Get expected dimension from first vector
        const expectedDimension = items[0].vector.length;
        if (expectedDimension === 0) {
            throw new Error('[Qdrant] First item has empty vector (dimension 0)');
        }

        // Validate all vectors have the same dimension
        const mismatchedItems = items.filter(item => item.vector.length !== expectedDimension);
        if (mismatchedItems.length > 0) {
            const examples = mismatchedItems.slice(0, 3).map(i => `hash ${i.hash}: ${i.vector.length}`).join(', ');
            throw new Error(`[Qdrant] Vector dimension mismatch. Expected ${expectedDimension}, but found: ${examples}${mismatchedItems.length > 3 ? '...' : ''} (${mismatchedItems.length} total mismatches)`);
        }

        // Never delete stored memories as a side effect of an embedding-model change.
        try {
            const info = await this._request('GET', `/collections/${mainCollection}`);
            const size = info.result?.config?.params?.vectors?.size;
            if (size && size !== expectedDimension) {
                throw new Error(`Vector dimension mismatch: stored ${size}, supplied ${expectedDimension}. Existing data preserved.`);
            }
        } catch (error) {
            if (!this._isNotFound(error)) throw error;
            await this.ensureCollection(mainCollection, expectedDimension);
        }

        // Format points for Qdrant with multitenancy payload
        // NOTE: Spread item.metadata FIRST so critical fields can override it
        // IMPORTANT: Ensure hash is a number - Qdrant accepts unsigned integers or UUID strings,
        // but NOT numeric strings like "977206". JSON parsing from HTTP requests can sometimes
        // convert numbers to strings, so we explicitly coerce here.
        const points = items.map(item => ({
            id: typeof item.hash === 'string' ? parseInt(item.hash, 10) : item.hash, // Ensure numeric ID for Qdrant
            vector: item.vector,
            payload: this._payload(item, tenantMetadata),
        }));

        // Upsert points (PUT with wait=true for reliability)
        await this._request('PUT', `/collections/${mainCollection}/points?wait=true`, {
            points: points,
        });

        console.log(`[Qdrant] Inserted ${items.length} vectors into ${mainCollection} (type: ${tenantMetadata.type}, sourceId: ${tenantMetadata.sourceId})`);
    }

    /** The stored payload of one item (shared by the legacy and v2 layouts). */
    _payload(item, tenantMetadata) {
        return {
                // ===== SPREAD ADDITIONAL METADATA FIRST (so it can be overridden) =====
                ...(item.metadata || {}),

                // ===== CORE FIELDS (override metadata) =====
                text: item.text,
                hash: typeof item.hash === 'string' ? parseInt(item.hash, 10) : item.hash, // Keep consistent with ID

                // ===== MULTITENANCY FIELDS (CRITICAL - must not be overwritten) =====
                type: tenantMetadata.type || 'chat',
                sourceId: tenantMetadata.sourceId || 'unknown',
                embeddingSource: tenantMetadata.embeddingSource || 'transformers',
                embeddingModel: tenantMetadata.embeddingModel || '',

                // ===== TIMESTAMPS (for temporal decay) =====
                timestamp: item.metadata?.timestamp || Date.now(),
                messageIndex: item.metadata?.messageIndex,

                // ===== LEGACY VECTHARE FEATURES =====
                importance: item.importance ?? item.metadata?.importance ?? 100,
                keywords: item.keywords || item.metadata?.keywords || [],
                customWeights: item.customWeights || item.metadata?.customWeights || {},
                disabledKeywords: item.disabledKeywords || item.metadata?.disabledKeywords || [],
                chunkGroup: item.chunkGroup || item.metadata?.chunkGroup || null,
                conditions: item.conditions || item.metadata?.conditions || null,
                summary: item.summary || item.metadata?.summary || null,
                isSummaryChunk: item.isSummaryChunk || item.metadata?.isSummaryChunk || false,
                parentHash: item.parentHash || item.metadata?.parentHash || null,

                // ===== CHAT-SPECIFIC =====
                speaker: item.metadata?.speaker,
                sceneTitle: item.metadata?.sceneTitle,
                sceneIndex: item.metadata?.sceneIndex,
                sceneStart: item.metadata?.sceneStart,
                sceneEnd: item.metadata?.sceneEnd,

                // ===== LOREBOOK-SPECIFIC =====
                entryName: item.metadata?.entryName,
                lorebookId: item.metadata?.lorebookId,

                // ===== CHARACTER-SPECIFIC =====
                characterName: item.metadata?.characterName,
                fieldName: item.metadata?.fieldName,

                // ===== DOCUMENT-SPECIFIC =====
                documentName: item.metadata?.documentName,
                url: item.metadata?.url,
                scrapeDate: item.metadata?.scrapeDate,
        };
    }

    /**
     * Query collection for similar vectors (MULTITENANCY)
     * @param {string} collectionName - Collection name (always "vecthare_main")
     * @param {number[]} queryVector - Query vector
     * @param {number} topK - Number of results
     * @param {object} filters - Payload filters {type, sourceId, minImportance, timestampAfter, etc.}
     * @returns {Promise<Array>} Results with {hash, text, score, metadata}
     */
    async queryCollection(collectionName, queryVector, topK = 10, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        try {
            // Check if collection exists
            const collections = await this._request('GET', '/collections');
            const exists = collections.result?.collections?.some(c => c.name === mainCollection);
            if (!exists) {
                return [];
            }

            // Build filter conditions
            const must = [];

            if (filters.type) {
                must.push({
                    key: 'type',
                    match: { value: filters.type }
                });
            }

            if (filters.sourceId) {
                must.push({
                    key: 'sourceId',
                    match: { value: filters.sourceId }
                });
            }

            if (filters.minImportance !== undefined) {
                must.push({
                    key: 'importance',
                    range: { gte: filters.minImportance }
                });
            }

            if (filters.timestampAfter !== undefined) {
                must.push({
                    key: 'timestamp',
                    range: { gte: filters.timestampAfter }
                });
            }

            if (filters.characterName) {
                must.push({
                    key: 'characterName',
                    match: { value: filters.characterName }
                });
            }

            if (filters.chatId) {
                must.push({
                    key: 'chatId',
                    match: { value: filters.chatId }
                });
            }

            if (filters.chunkGroup) {
                must.push({
                    key: 'chunkGroup.name',
                    match: { value: filters.chunkGroup }
                });
            }

            if (filters.embeddingSource) {
                must.push({
                    key: 'embeddingSource',
                    match: { value: filters.embeddingSource }
                });
            }
            // 9c: v2 tenant scope / explicit legacy content_type scope.
            must.push(...this._extraScope(filters));

            // Build search payload
            const searchPayload = {
                vector: queryVector,
                limit: topK,
                with_payload: true,
            };

            // Add filters if any
            if (must.length > 0) {
                searchPayload.filter = { must };
            }

            // Search
            const response = await this._request('POST', `/collections/${mainCollection}/points/search`, searchPayload, 1, 20000); // VH-019

            // Format results
            return (response.result || []).map(result => ({
                hash: result.payload.hash,
                text: result.payload.text,
                score: result.score,
                metadata: result.payload,
            }));
        } catch (error) {
            if (this._isNotFound(error)) return [];
            console.error(`[Qdrant] Query failed for ${mainCollection}:`, error.message);
            throw error;
        }
    }

    /**
     * Native hybrid query combining vector similarity and keyword matching (QDRANT OPTIMIZED)
     * Uses Qdrant's built-in full-text search capabilities for better performance
     * @param {string} collectionName - Collection name (always "vecthare_main")
     * @param {number[]} queryVector - Query vector for semantic search
     * @param {string[]} keywords - Keywords for full-text matching (BM25-like)
     * @param {number} topK - Number of results to return
     * @param {object} options - Query options
     *   - vectorWeight: Weight for vector similarity (0-1, default: 0.5)
     *   - keywordWeight: Weight for keyword matching (0-1, default: 0.5)
     *   - fusionMethod: 'rrf' or 'weighted' (default: 'rrf')
     *   - rrfK: RRF constant (default: 60)
     *   - keywordBoost: Multiplier for keyword matches (default: 1.5)
     * @param {object} filters - Payload filters {type, sourceId, etc.}
     * @returns {Promise<Array>} Hybrid search results with {hash, text, score, metadata, debug}
     */
    async hybridQuery(collectionName, queryVector, keywords = [], topK = 10, options = {}, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        console.log(`[Qdrant] Hybrid query START: collection=${mainCollection}, keywords=${keywords.length}, topK=${topK}`);

        // Default options
        const vectorWeight = options.vectorWeight ?? 0.5;
        const keywordWeight = options.keywordWeight ?? options.textWeight ?? 0.5;
        const fusionMethod = options.fusionMethod || 'rrf';
        const rrfK = options.rrfK || 60;
        const keywordBoost = options.keywordBoost || 1.5;

        console.log(`[Qdrant] Hybrid options: vectorWeight=${vectorWeight}, keywordWeight=${keywordWeight}, fusion=${fusionMethod}`);

        try {
            // Check if collection exists
            const collections = await this._request('GET', '/collections');
            const exists = collections.result?.collections?.some(c => c.name === mainCollection);
            if (!exists) {
                console.warn(`[Qdrant] Hybrid query ABORT: Collection ${mainCollection} does not exist`);
                return [];
            }

            console.log(`[Qdrant] Collection ${mainCollection} exists, proceeding with hybrid search`);

            // Build base filter conditions
            const must = [];

            if (filters.type) {
                must.push({
                    key: 'type',
                    match: { value: filters.type }
                });
            }

            if (filters.sourceId) {
                must.push({
                    key: 'sourceId',
                    match: { value: filters.sourceId }
                });
            }

            if (filters.minImportance !== undefined) {
                must.push({
                    key: 'importance',
                    range: { gte: filters.minImportance }
                });
            }

            if (filters.timestampAfter !== undefined) {
                must.push({
                    key: 'timestamp',
                    range: { gte: filters.timestampAfter }
                });
            }

            if (filters.characterName) {
                must.push({
                    key: 'characterName',
                    match: { value: filters.characterName }
                });
            }

            if (filters.chatId) {
                must.push({
                    key: 'chatId',
                    match: { value: filters.chatId }
                });
            }

            if (filters.chunkGroup) {
                must.push({
                    key: 'chunkGroup.name',
                    match: { value: filters.chunkGroup }
                });
            }

            if (filters.embeddingSource) {
                must.push({
                    key: 'embeddingSource',
                    match: { value: filters.embeddingSource }
                });
            }
            // 9c: v2 tenant scope / explicit legacy content_type scope.
            must.push(...this._extraScope(filters));

            // ================================================================
            // STRATEGY 1: Vector Search (Semantic Similarity)
            // ================================================================
            const vectorPayload = {
                vector: queryVector,
                limit: topK * 2, // Fetch more for fusion
                with_payload: true,
            };

            if (must.length > 0) {
                vectorPayload.filter = { must: [...must] };
            }

            const vectorResponse = await this._request('POST', `/collections/${mainCollection}/points/search`, vectorPayload, 1, 20000); // VH-019
            const vectorResults = (vectorResponse.result || []).map(result => ({
                hash: result.payload.hash,
                text: result.payload.text,
                score: result.score,
                vectorScore: result.score,
                metadata: result.payload,
            }));

            console.log(`[Qdrant] Vector search complete: ${vectorResults.length} results`);

            // ================================================================
            // STRATEGY 2: Keyword Search (Full-Text Matching)
            // ================================================================
            let keywordResults = [];
            let keywordSearchTruncated = false;

            if (keywords && keywords.length > 0) {
                console.log(`[Qdrant] Starting keyword search with ${keywords.length} keywords: ${keywords.join(', ')}`);

                // Build keyword filter using text matching on 'text' and 'keywords' payload fields
                const keywordConditions = [];

                for (const keyword of keywords) {
                    // Search in text field (full-text)
                    keywordConditions.push({
                        key: 'text',
                        match: { text: keyword }
                    });

                    // Search in keywords array field (stored keywords)
                    keywordConditions.push({
                        key: 'keywords',
                        match: { any: [keyword] }
                    });
                    // Qdrant text-match uses the case-insensitive keyword-object index.
                    keywordConditions.push({
                        key: 'keywords[].text',
                        match: { text: keyword }
                    });
                }

                // Scroll through points matching keyword filter
                let offset = null;
                const seenOffsets = new Set();
                const maxKeywordCandidates = 20000;
                const keywordPageSize = 500;
                let candidatesExamined = 0;

                do {
                    const scrollPayload = {
                        limit: Math.min(keywordPageSize, maxKeywordCandidates - candidatesExamined),
                        // Full payload is needed by later RAG metadata/conditions.
                        with_payload: true,
                        with_vector: false,
                        filter: {
                            must: [...must],
                            should: keywordConditions,
                        }
                    };

                    if (offset !== null) {
                        scrollPayload.offset = offset;
                    }

                    let scrollResponse;
                    try {
                        scrollResponse = await this._request('POST', `/collections/${mainCollection}/points/scroll`, scrollPayload);
                    } catch (error) {
                        const missingIndex = (error?.status === 400 || error?.status === 422)
                            && /(?:index|unindexed|strict.mode)/i.test(String(error.message));
                        if (!missingIndex) throw error;
                        console.warn(`[Qdrant] Keyword index unavailable for ${mainCollection}; using vector-only results:`, error.message);
                        keywordResults = [];
                        keywordSearchTruncated = false;
                        break;
                    }
                    const points = scrollResponse.result?.points || [];
                    candidatesExamined += points.length;

                    // Calculate keyword match scores for each point
                    for (const point of points) {
                        const text = (point.payload.text || '').toLowerCase();
                        const storedKeywords = (point.payload.keywords || []).map(k =>
                            typeof k === 'string' ? k.toLowerCase() : (k.text || '').toLowerCase()
                        );

                        let matchCount = 0;
                        for (const keyword of keywords) {
                            const kw = keyword.toLowerCase();
                            // Check text content
                            if (text.includes(kw)) matchCount++;
                            // Check stored keywords
                            if (storedKeywords.some(sk => sk.includes(kw) || kw.includes(sk))) matchCount++;
                        }

                        if (matchCount > 0) {
                            const keywordScore = Math.min(1.0, (matchCount / keywords.length) * keywordBoost);
                            keywordResults.push({
                                hash: point.payload.hash,
                                text: point.payload.text,
                                keywordScore: keywordScore,
                                matchedKeywords: matchCount,
                                metadata: point.payload,
                            });
                        }
                    }

                    offset = scrollResponse.result?.next_page_offset;
                    if (offset !== null && offset !== undefined) {
                        const offsetKey = JSON.stringify(offset);
                        if (seenOffsets.has(offsetKey)) {
                            keywordSearchTruncated = true;
                            console.warn('[Qdrant] Keyword scroll repeated an offset; keeping results gathered so far');
                            break;
                        }
                        seenOffsets.add(offsetKey);
                    }
                    if (candidatesExamined >= maxKeywordCandidates && offset != null) {
                        keywordSearchTruncated = true;
                        console.warn(`[Qdrant] Keyword search capped at ${maxKeywordCandidates} matching candidates`);
                        break;
                    }

                } while (offset !== null && offset !== undefined);

                const matchCount = result => Array.isArray(result.matchedKeywords)
                    ? result.matchedKeywords.length
                    : (result.matchedKeywords || 0);
                keywordResults.sort((a, b) =>
                    ((b.keywordScore ?? 0) - (a.keywordScore ?? 0)) ||
                    (matchCount(b) - matchCount(a))
                );
                console.log(`[Qdrant] Keyword search complete: ${keywordResults.length} results`);
            } else {
                console.log(`[Qdrant] No keywords provided, skipping keyword search`);
            }

            // ================================================================
            // STRATEGY 3: Fusion (Combine Vector + Keyword Results)
            // ================================================================
            console.log(`[Qdrant] Starting fusion: ${vectorResults.length} vector + ${keywordResults.length} keyword results`);

            const fusedResults = this._fuseResults(vectorResults, keywordResults, {
                method: fusionMethod,
                vectorWeight,
                keywordWeight,
                rrfK,
                topK,
            });
            if (keywordSearchTruncated) {
                for (const result of fusedResults) result.debug.keywordSearchTruncated = true;
            }

            console.log(`[Qdrant] Hybrid query: ${vectorResults.length} vector + ${keywordResults.length} keyword = ${fusedResults.length} fused results`);

            return fusedResults;

        } catch (error) {
            if (this._isNotFound(error)) return [];
            console.error(`[Qdrant] Hybrid query failed for ${mainCollection}:`, error.message);
            throw error;
        }
    }

    /**
     * Fuse vector and keyword search results using RRF or weighted combination
     * @param {Array} vectorResults - Results from vector search
     * @param {Array} keywordResults - Results from keyword search
     * @param {object} options - Fusion options {method, vectorWeight, keywordWeight, rrfK, topK}
     * @returns {Array} Fused results sorted by combined score
     */
    _fuseResults(vectorResults, keywordResults, options) {
        const { method, vectorWeight, keywordWeight, rrfK, topK } = options;

        // Create hash map for merging results
        const resultsMap = new Map();

        // Add vector results
        vectorResults.forEach((result, index) => {
            resultsMap.set(result.hash, {
                ...result,
                vectorRank: index + 1,
                vectorScore: result.vectorScore ?? result.score,
                keywordScore: 0,
                keywordRank: Infinity,
            });
        });

        // Merge keyword results
        keywordResults.forEach((result, index) => {
            if (resultsMap.has(result.hash)) {
                const existing = resultsMap.get(result.hash);
                existing.keywordScore = result.keywordScore;
                existing.keywordRank = index + 1;
                existing.matchedKeywords = result.matchedKeywords;
            } else {
                resultsMap.set(result.hash, {
                    ...result,
                    vectorScore: 0,
                    vectorRank: Infinity,
                    keywordScore: result.keywordScore,
                    keywordRank: index + 1,
                });
            }
        });

        // RRF determines ranking; score remains on the 0–1 relevance scale used by RAG.
        const fusedResults = Array.from(resultsMap.values()).map(result => {
            let rankScore;
            let relevanceScore;
            if (method === 'rrf') {
                // Missing channels use Infinity ranks, never 0 (see resultsMap above).
                const hasVector = Number.isFinite(result.vectorRank);
                const hasKeyword = Number.isFinite(result.keywordRank);
                rankScore = (vectorWeight * (hasVector ? 1 / (rrfK + result.vectorRank) : 0))
                    + (keywordWeight * (hasKeyword ? 1 / (rrfK + result.keywordRank) : 0));

                const vectorSimilarity = Math.max(0, Math.min(1, Number(result.vectorScore) || 0));
                const keywordRelevance = Math.max(0, Math.min(1, Number(result.keywordScore) || 0));
                const presentVectorWeight = hasVector ? vectorWeight : 0;
                const presentKeywordWeight = hasKeyword ? keywordWeight : 0;
                const presentWeight = presentVectorWeight + presentKeywordWeight;
                // Keyword score and cosine similarity are useful evidence, not calibrated probabilities.
                relevanceScore = presentWeight > 0
                    ? (presentVectorWeight * vectorSimilarity + presentKeywordWeight * keywordRelevance) / presentWeight
                    : 0;
            } else {
                // Preserve the existing weighted-fusion calculation and ordering.
                rankScore = (vectorWeight * result.vectorScore) + (keywordWeight * result.keywordScore);
                relevanceScore = rankScore;
            }

            return {
                hash: result.hash,
                text: result.text,
                score: relevanceScore,
                metadata: result.metadata,
                debug: {
                    vectorScore: result.vectorScore,
                    keywordScore: result.keywordScore,
                    vectorRank: result.vectorRank,
                    keywordRank: result.keywordRank,
                    matchedKeywords: result.matchedKeywords || 0,
                    fusionMethod: method,
                    rankScore,
                },
            };
        });

        const compareFused = method === 'rrf'
            ? (a, b) => (b.debug.rankScore - a.debug.rankScore)
                || (b.score - a.score)
                || String(a.hash).localeCompare(String(b.hash))
            : (a, b) => (b.score - a.score)
                || String(a.hash).localeCompare(String(b.hash));
        return fusedResults.sort(compareFused).slice(0, topK);
    }

    /**
     * List all items in a collection (MULTITENANCY)
     * @param {string} collectionName - Collection name (always "vecthare_main")
     * @param {object} filters - Payload filters {type, sourceId}
     * @param {object} options - Options { includeVectors }
     * @returns {Promise<Array>} Array of items with {hash, text, metadata, vector?}
     */
    async listItems(collectionName, filters = {}, options = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        // MULTITENANCY: Always use vecthare_main collection
        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        try {
            // Check if collection exists
            const collections = await this._request('GET', '/collections');
            const exists = collections.result?.collections?.some(c => c.name === mainCollection);
            if (!exists) {
                return [];
            }

            // Build filter conditions
            const must = [];
            if (filters.type) {
                must.push({
                    key: 'type',
                    match: { value: filters.type }
                });
            }
            if (filters.sourceId) {
                must.push({
                    key: 'sourceId',
                    match: { value: filters.sourceId }
                });
            }
            must.push(...this._extraScope(filters));

            // Scroll through all points
            const items = [];
            let offset = null;

            do {
                const scrollPayload = {
                    limit: options.hashesOnly ? 1000 : 100,
                    with_payload: options.hashesOnly ? { include: ['hash'] } : true,
                    with_vector: !options.hashesOnly && Boolean(options.includeVectors),
                };

                if (offset !== null) {
                    scrollPayload.offset = offset;
                }

                // Add filters if any
                if (must.length > 0) {
                    scrollPayload.filter = { must };
                }

                const response = await this._request('POST', `/collections/${mainCollection}/points/scroll`, scrollPayload);
                // A success-shaped reply without a points array is not an empty
                // final page; never let it end the listing as "complete".
                if (!Array.isArray(response?.result?.points)) {
                    throw new Error(`Invalid Qdrant scroll response for ${mainCollection}`);
                }
                if (options.hashesOnly && (!Array.isArray(response?.result?.points) ||
                    response.result.points.some(point => point?.payload?.hash === undefined ||
                        point.payload.hash === null))) {
                    throw new Error(`Invalid Qdrant hash-only scroll response for ${mainCollection}`);
                }

                items.push(...(response.result?.points || []).map(p =>
                    options.hashesOnly ? { hash: p.payload.hash } : {
                        hash: p.payload.hash,
                        text: p.payload.text,
                        metadata: p.payload,
                        vector: options.includeVectors ? p.vector : undefined,
                    }
                ));
                offset = response.result?.next_page_offset;
            } while (offset !== null && offset !== undefined);

            if (options.strictRead) {
                // Independent completeness evidence: Qdrant's own exact count for
                // the same filter must equal what the scroll returned.
                const counted = await this._request('POST', `/collections/${mainCollection}/points/count`, {
                    exact: true, ...(must.length > 0 ? { filter: { must } } : {}),
                });
                if (!Number.isSafeInteger(counted?.result?.count) || counted.result.count !== items.length) {
                    throw new Error(`Incomplete Qdrant listing for ${mainCollection}: scrolled ${items.length}, collection reports ${counted?.result?.count}`);
                }
            }
            console.log(">>> [Qdrant] Listed", items.length, "items from", mainCollection);
            return items;
        } catch (error) {
            if (this._isNotFound(error)) return [];
            console.error(`[Qdrant] Failed to list items from ${mainCollection}:`, error.message);
            throw error;
        }
    }

    /**
     * Get all saved hashes from a collection (MULTITENANCY)
     * @param {string} collectionName - Collection name (always "vecthare_main")
     * @param {object} filters - Payload filters {type, sourceId}
     * @returns {Promise<number[]>} Array of hashes
     */
    async getSavedHashes(collectionName, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        try {
            // Check if collection exists
            const collections = await this._request('GET', '/collections');
            const exists = collections.result?.collections?.some(c => c.name === mainCollection);
            if (!exists) {
                return [];
            }

            // Build filter conditions
            const must = [];
            if (filters.type) {
                must.push({
                    key: 'type',
                    match: { value: filters.type }
                });
            }
            if (filters.sourceId) {
                must.push({
                    key: 'sourceId',
                    match: { value: filters.sourceId }
                });
            }
            must.push(...this._extraScope(filters));

            // Scroll through all points to get hashes
            const hashes = [];
            let offset = null;

            do {
                const scrollPayload = {
                    limit: 100,
                    with_payload: { include: ['hash'] },
                    with_vector: false,
                };

                if (offset !== null) {
                    scrollPayload.offset = offset;
                }

                // Add filters if any
                if (must.length > 0) {
                    scrollPayload.filter = { must };
                }

                const response = await this._request('POST', `/collections/${mainCollection}/points/scroll`, scrollPayload);

                hashes.push(...(response.result?.points || []).map(p => p.payload.hash));
                offset = response.result?.next_page_offset;
            } while (offset !== null && offset !== undefined);

            return hashes;
        } catch (error) {
            console.error(`[Qdrant] Failed to get hashes from ${mainCollection}:`, error.message);
            return [];
        }
    }

    /**
     * Delete specific items by hash (MULTITENANCY)
     * @param {string} collectionName - Collection name (always "vecthare_main")
     * @param {number[]} hashes - Hashes to delete
     * @returns {Promise<void>}
     */
    async deleteVectors(collectionName, hashes, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        if (hashes.length === 0) return;

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        // 9a fix: the legacy shared store keeps every collection's points side by
        // side, and a point id is only the text hash. A delete by id therefore
        // removed whichever collection's point carried that hash - e.g. Sync's
        // ghost cleanup in one chat deleting another chat's vectors. Deletes in
        // that store are now limited to the caller's own points (content_type),
        // and refused when the caller does not say which collection it is.
        let legacyScope = null;
        if (mainCollection === 'vecthare_multitenancy') {
            legacyScope = filters?.content_type;
            if (!isLegacyContentType(legacyScope)) {
                const error = new Error('Deleting from the shared Qdrant store needs the collection it belongs to; nothing was deleted.');
                error.status = 409;
                throw error;
            }
        }

        try {
            // Delete points by ID (hash)
            // Ensure hashes are numbers - Qdrant requires unsigned integers or UUID strings
            const numericHashes = hashes.map(h => typeof h === 'string' ? parseInt(h, 10) : h);
            await this._request('POST', `/collections/${mainCollection}/points/delete?wait=true`,
                legacyScope === null ? { points: numericHashes } : {
                    filter: { must: [
                        { has_id: numericHashes },
                        { key: 'content_type', match: { value: legacyScope } },
                    ] },
                });

            console.log(`[Qdrant] Deleted ${hashes.length} items from ${mainCollection}`);
        } catch (error) {
            if (this._isNotFound(error)) return;
            console.error(`[Qdrant] Delete failed for ${mainCollection}:`, error.message);
            throw error;
        }
    }

    /**
     * Purge collection for a specific source (MULTITENANCY)
     * Deletes all points matching type and sourceId filters
     * @param {string} collectionName - Collection name (always "vecthare_main")
     * @param {object} filters - Payload filters {type, sourceId}
     * @returns {Promise<void>}
     */
    async purgeCollection(collectionName, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        try {
            // Check if collection exists
            const collections = await this._request('GET', '/collections');
            const exists = collections.result?.collections?.some(c => c.name === mainCollection);
            if (!exists) {
                return;
            }

            // Build filter conditions
            const must = [];
            if (filters.type) {
                must.push({
                    key: 'type',
                    match: { value: filters.type }
                });
            }
            if (filters.sourceId) {
                must.push({
                    key: 'sourceId',
                    match: { value: filters.sourceId }
                });
            }

            if (must.length === 0) {
                console.warn('[Qdrant] No filters provided to purgeCollection - use purgeAll() instead');
                return;
            }

            // Delete points by filter
            await this._request('POST', `/collections/${mainCollection}/points/delete?wait=true`, {
                filter: { must }
            });

            console.log(`[Qdrant] Purged ${mainCollection} (type: ${filters.type}, sourceId: ${filters.sourceId})`);
        } catch (error) {
            if (this._isNotFound(error)) {
                return;
            }
            console.error(`[Qdrant] Purge failed for ${mainCollection}:`, error.message);
            throw error;
        }
    }

    /**
     * Purge entire vecthare_main collection (MULTITENANCY)
     * WARNING: Deletes ALL data from ALL sources
     * @returns {Promise<void>}
     */
    async purgeAll(collectionName) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;
        // H2: a shared physical store holds many VectHare collections; no route
        // may drop it. One wrong id here used to delete every chat's vectors.
        if (isReservedSharedCollection(mainCollection)) {
            const error = new Error(`Refusing to delete shared store "${mainCollection}": it holds several collections. ` +
                'Delete individual collections instead; nothing was changed.');
            error.status = 409;
            throw error;
        }

        try {
            await this._request('DELETE', `/collections/${mainCollection}`);
            console.log(`[Qdrant] Purged entire collection: ${mainCollection}`);
        } catch (error) {
            if (this._isNotFound(error)) {
                return;
            }
            console.error(`[Qdrant] Purge all failed:`, error.message);
            throw error;
        }
    }

    // ========================================================================
    // ADDITIONAL METHODS REQUIRED BY PLUGIN ROUTER
    // ========================================================================



    /**
     * Explicit existence of one collection, for the Database Browser's
     * classification. Errors propagate: an unreachable Qdrant must be reported
     * as unverified by the caller, never as absence.
     * @param {string} collectionName Physical collection name
     * @returns {Promise<boolean>} Whether Qdrant reports the collection
     */
    async collectionExists(collectionName) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        const collections = await this._request('GET', '/collections');
        const names = collections.result?.collections?.map(item => item.name);
        if (!Array.isArray(names)) throw new Error('Qdrant returned no collection list');
        return names.includes(collectionName);
    }

    // ========================================================================
    // 9c: MULTITENANCY v2 (vecthare_mt_<dim>)
    // ========================================================================
    // One shared collection per vector size. Every operation is scoped by the
    // plugin-computed tenantKey; a point id is UUIDv5(tenantKey:hash), so the
    // same text in two collections never shares a point and re-inserting a
    // chunk is idempotent. The physical collection is never dropped (H2).

    /** Creates the v2 collection (vector size `dim`) and its payload indexes when missing. */
    async ensureV2Collection(name, dim) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        if (!MT_V2_NAME.test(name)) throw new Error(`Not a v2 collection name: ${name}`);
        // The name carries the vector size, so a store checked once stays valid
        // (it is never dropped); skip the two extra reads on later inserts.
        this.v2Ensured ??= new Set();
        const memo = `${this.baseUrl}\u0000${name}`;
        if (this.v2Ensured.has(memo) && Number(MT_V2_NAME.exec(name)[1]) === dim) return;
        let info = null;
        try {
            info = await this._request('GET', `/collections/${name}`);
        } catch (error) {
            if (!this._isNotFound(error)) throw error;
        }
        if (info) {
            const size = info.result?.config?.params?.vectors?.size;
            if (size && size !== dim) {
                const error = new Error(`[Qdrant] ${name} holds ${size}-dim vectors; this model makes ${dim}. Nothing was changed.`);
                error.status = 409;
                throw error;
            }
            await this._ensureV2Indexes(name);
            this.v2Ensured.add(memo);
            return;
        }
        await this._request('PUT', `/collections/${name}`, { vectors: { size: dim, distance: 'Cosine' } });
        console.log(`[Qdrant] Created v2 collection: ${name} (dim=${dim})`);
        await this._ensureV2Indexes(name);
        this.v2Ensured.add(memo);
    }

    async _ensureV2Indexes(name) {
        const info = await this._request('GET', `/collections/${name}`);
        const existing = info.result?.payload_schema || {};
        const wanted = [
            { field: 'tenantKey', schema: { type: 'keyword', is_tenant: true } },
            { field: 'hash', schema: 'integer' },
            { field: 'embeddingSource', schema: 'keyword' },
            { field: 'keywords', schema: 'keyword' },
            { field: 'keywords[].text', schema: { type: 'text', tokenizer: 'word', lowercase: true } },
            { field: 'text', schema: 'text' },
            { field: 'timestamp', schema: 'integer' },
            { field: 'importance', schema: 'integer' },
        ];
        for (const { field, schema } of wanted) {
            const type = typeof schema === 'string' ? schema : schema.type;
            if (existing[field]?.data_type === type) continue;
            try {
                await this._request('PUT', `/collections/${name}/index?wait=true`, { field_name: field, field_schema: schema });
            } catch (error) {
                if (!/already exists|409/.test(String(error.message))) {
                    console.warn(`[Qdrant] Failed to create index ${field} on ${name}:`, error.message);
                }
            }
        }
    }

    /** Point id of one chunk of one tenant. */
    v2PointId(tenantKey, hash) {
        return uuidv5(`${tenantKey}:${hash}`);
    }

    /** Insert/upsert into a v2 collection. `tenant` = {tenantKey, user, type, sourceId}. */
    async insertV2(name, items, tenant, embedding = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        if (!items.length) return;
        if (items.some(item => !Array.isArray(item.vector) || !item.vector.length)) {
            throw new Error('[Qdrant] Items are missing vectors');
        }
        const dim = items[0].vector.length;
        if (items.some(item => item.vector.length !== dim)) throw new Error('[Qdrant] Vector dimension mismatch inside one batch');
        await this.ensureV2Collection(name, dim);
        const points = items.map(item => {
            const hash = typeof item.hash === 'string' ? parseInt(item.hash, 10) : item.hash;
            if (!Number.isSafeInteger(hash)) throw new Error('[Qdrant] Invalid chunk hash');
            const payload = this._payload({ ...item, hash }, {
                type: tenant.type, sourceId: tenant.sourceId,
                embeddingSource: embedding.embeddingSource, embeddingModel: embedding.embeddingModel,
            });
            delete payload.content_type;
            payload.tenantKey = tenant.tenantKey;
            payload.user = tenant.user;
            return { id: this.v2PointId(tenant.tenantKey, hash), vector: item.vector, payload };
        });
        await this._request('PUT', `/collections/${name}/points?wait=true`, { points });
    }

    /** Exact number of points of one tenant. */
    async countTenant(name, tenantKey, extraMust = []) {
        const res = await this._request('POST', `/collections/${name}/points/count`, {
            exact: true, filter: { must: [{ key: 'tenantKey', match: { value: tenantKey } }, ...extraMust] },
        });
        const count = res?.result?.count;
        if (!Number.isSafeInteger(count) || count < 0) throw new Error(`Invalid Qdrant count for ${name}`);
        return count;
    }

    /** Deletes this tenant's points for the given hashes (never another tenant's). */
    async deleteV2(name, hashes, tenantKey) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        const list = hashes.map(h => typeof h === 'string' ? parseInt(h, 10) : h).filter(Number.isSafeInteger);
        if (!list.length) return 0;
        try {
            for (let i = 0; i < list.length; i += 1000) {
                await this._request('POST', `/collections/${name}/points/delete?wait=true`, {
                    filter: { must: [
                        { key: 'tenantKey', match: { value: tenantKey } },
                        { key: 'hash', match: { any: list.slice(i, i + 1000) } },
                    ] },
                });
            }
        } catch (error) {
            if (this._isNotFound(error)) return 0;
            throw error;
        }
        return list.length;
    }

    /**
     * Removes all of one tenant's points, confirms none remain, and empties the
     * tenant's saved-settings slot (a tombstone, so a later first write can
     * record new ones). The shared collection itself stays.
     */
    async purgeV2(name, tenantKey, configKey) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        try {
            await this._request('POST', `/collections/${name}/points/delete?wait=true`, {
                filter: { must: [{ key: 'tenantKey', match: { value: tenantKey } }] },
            });
        } catch (error) {
            if (this._isNotFound(error)) return;
            throw error;
        }
        const left = await this.countTenant(name, tenantKey);
        if (left !== 0) throw new Error(`[Qdrant] Purge could not be confirmed: ${left} point(s) remain`);
        if (configKey) {
            const current = (await this._request('GET', `/collections/${name}`))?.result?.config?.metadata?.[configKey];
            if (current !== undefined && !(current && current.vecthareTombstone === 1)) {
                await this._request('PATCH', `/collections/${name}`, { metadata: { [configKey]: { vecthareTombstone: 1 } } });
            }
        }
    }

    /** v2 text/metadata edit: same point id for the same hash; a new hash gets a new point. */
    async updateV2(name, hash, updates, tenant, embedding = {}) {
        const existing = await this.getItem(name, hash, { tenantKey: tenant.tenantKey });
        if (!existing) throw new Error(`Item with hash ${hash} not found`);
        const newHash = updates.hash || hash;
        const metadata = mergeRowMetadata(existing.metadata, updates);
        delete metadata.tenantKey; delete metadata.user;
        await this.insertV2(name, [{ hash: newHash, text: updates.text || existing.text,
            vector: updates.vector || existing.vector, metadata }], tenant, {
            embeddingSource: embedding.embeddingSource ?? existing.metadata?.embeddingSource,
            embeddingModel: embedding.embeddingModel ?? existing.metadata?.embeddingModel,
        });
        if (String(newHash) !== String(hash)) await this.deleteV2(name, [hash], tenant.tenantKey);
    }

    /** v2 collections (by name), for locating a tenant whose vector size is not known. */
    async listV2CollectionNames() {
        return (await this.listCollectionNames()).filter(name => MT_V2_NAME.test(name));
    }

    /** 9a: collection names; errors propagate (never "no collections"). */
    async listCollectionNames() {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        const collections = await this._request('GET', '/collections');
        const names = collections?.result?.collections?.map(item => item?.name);
        if (!Array.isArray(names) || names.some(name => typeof name !== 'string')) {
            throw new Error('Qdrant returned no collection list');
        }
        return names;
    }

    async getCollections() {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        try {
            const collections = await this._request('GET', '/collections');
            return collections.result?.collections?.map(c => c.name) || [];
        } catch (error) {
            console.error(`[Qdrant] getCollections failed:`, error.message);
            return [];
        }
    }

    /**
     * Get a single item by hash (MULTITENANCY)
     * @param {string} collectionName - Collection name (always "vecthare_main")
     * @param {number} hash - Item hash to find
     * @param {object} filters - Payload filters {type, sourceId}
     * @returns {Promise<object|null>} Item or null if not found
     */
    async getItem(collectionName, hash, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;
        console.log(mainCollection);
        // Ensure hash is numeric for consistent matching with stored payload
        const numericHash = typeof hash === 'string' ? parseInt(hash, 10) : hash;

        try {
            // Build filter to find specific item
            const must = [
                { key: 'hash', match: { value: numericHash } }
            ];

            if (filters.type) {
                must.push({ key: 'type', match: { value: filters.type } });
            }
            if (filters.sourceId) {
                must.push({ key: 'sourceId', match: { value: filters.sourceId } });
            }
            must.push(...this._extraScope(filters));

            const response = await this._request('POST', `/collections/${mainCollection}/points/scroll`, {
                filter: { must },
                limit: 1,
                with_payload: true,
                with_vector: true,
            });

            const points = response.result?.points || [];
            if (points.length === 0) {
                return null;
            }

            const p = points[0];
            return {
                hash: p.payload.hash,
                text: p.payload.text,
                vector: p.vector,
                metadata: p.payload,
            };
        } catch (error) {
            if (this._isNotFound(error)) return null;
            console.error(`[Qdrant] getItem failed:`, error.message);
            throw error;
        }
    }

    /**
     * Update an item (delete and re-insert with new data)
     * @param {string} collectionName - Collection name
     * @param {number} hash - Item hash to update
     * @param {object} updates - Updated fields {text?, hash?, vector?, ...metadata}
     * @param {object} filters - Multitenancy filters {type, sourceId}
     * @returns {Promise<void>}
     */
    async updateItem(collectionName, hash, updates, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        // Get existing item
        const existing = await this.getItem(collectionName, hash, filters);
        if (!existing) {
            throw new Error(`Item with hash ${hash} not found`);
        }

        // Merge updates with existing data
        const newHash = updates.hash || hash;
        const newItem = {
            hash: newHash,
            text: updates.text || existing.text,
            vector: updates.vector || existing.vector,
            metadata: mergeRowMetadata(existing.metadata, updates),
        };

        // Write the replacement first (a point's ID is its hash, so an upsert
        // replaces in place); remove the old hash only if it actually changed.
        await this.insertVectors(collectionName, [newItem], filters);
        if (String(newHash) !== String(hash)) await this.deleteVectors(collectionName, [hash], filters);

        console.log(`[Qdrant] Updated item ${hash} -> ${newHash}`);
    }

    /**
     * Update item metadata only (no re-embedding needed)
     * @param {string} collectionName - Collection name
     * @param {number} hash - Item hash to update
     * @param {object} metadata - New metadata fields to merge
     * @param {object} filters - Multitenancy filters {type, sourceId}
     * @returns {Promise<void>}
     */
    async updateItemMetadata(collectionName, hash, metadata, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        // Get existing item (need the vector)
        const existing = await this.getItem(collectionName, hash, filters);
        if (!existing) {
            throw new Error(`Item with hash ${hash} not found`);
        }

        // Create updated item with same text/vector but new metadata
        const updatedItem = {
            hash: hash,
            text: existing.text,
            vector: existing.vector,
            metadata: { ...existing.metadata, ...metadata },
        };

        // Same hash = same point ID: the upsert replaces it in place.
        await this.insertVectors(collectionName, [updatedItem], filters);

        console.log(`[Qdrant] Updated metadata for item ${hash}`);
    }

    /**
     * Query vectors with threshold filtering (alias for queryCollection)
     * @param {string} collectionName - Collection name
     * @param {number[]} queryVector - Query vector
     * @param {number} topK - Number of results
     * @param {number} threshold - Minimum score threshold
     * @param {object} filters - Payload filters
     * @returns {Promise<Array>} Results above threshold
     */
    async queryVectors(collectionName, queryVector, topK, threshold, filters = {}) {
        const results = await this.queryCollection(collectionName, queryVector, topK, filters);
        // Filter by threshold
        return results.filter(r => r.score >= threshold);
    }

    /**
     * Get collection statistics (MULTITENANCY)
     * @param {string} collectionName - Collection name
     * @param {object} filters - Payload filters {type, sourceId}
     * @returns {Promise<object>} Statistics object
     */
    async getCollectionStats(collectionName, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        try {
            // Check if collection exists
            const collections = await this._request('GET', '/collections');
            const exists = collections.result?.collections?.some(c => c.name === mainCollection);
            if (!exists) {
                return {
                    chunkCount: 0,
                    totalCharacters: 0,
                    totalTokens: 0,
                    storageSize: 0,
                    embeddingDimensions: 0,
                    avgChunkSize: 0,
                    messageCount: 0,
                    sources: {},
                    backend: 'qdrant',
                };
            }

            // Get collection info from Qdrant
            const collectionInfo = await this._request('GET', `/collections/${mainCollection}`);
            const totalPoints = collectionInfo.result?.points_count || 0;
            const vectorSize = collectionInfo.result?.config?.params?.vectors?.size || 0;

            // Get items matching filters to calculate stats
            const items = await this.listItems(collectionName, filters, { includeVectors: false });

            let totalCharacters = 0;
            let totalTokens = 0;
            const sources = {};
            const messageHashes = new Set();

            for (const item of items) {
                const text = item.text || '';
                totalCharacters += text.length;
                totalTokens += Math.ceil(text.length / 4); // Rough estimate

                const src = item.metadata?.embeddingSource || 'unknown';
                sources[src] = (sources[src] || 0) + 1;

                if (item.metadata?.originalMessageHash) {
                    messageHashes.add(item.metadata.originalMessageHash);
                }
            }

            return {
                chunkCount: items.length,
                totalPoints: totalPoints, // Total in collection (all tenants)
                totalCharacters,
                totalTokens,
                storageSize: 0, // Qdrant doesn't expose this easily
                embeddingDimensions: vectorSize,
                avgChunkSize: items.length > 0 ? Math.round(totalCharacters / items.length) : 0,
                messageCount: messageHashes.size,
                sources,
                backend: 'qdrant',
            };
        } catch (error) {
            console.error(`[Qdrant] getCollectionStats failed:`, error.message);
            return {
                chunkCount: 0,
                totalCharacters: 0,
                totalTokens: 0,
                storageSize: 0,
                embeddingDimensions: 0,
                avgChunkSize: 0,
                messageCount: 0,
                sources: {},
                backend: 'qdrant',
                error: error.message,
            };
        }
    }

    /**
     * Check if a chunk with given message IDs already exists (duplicate detection)
     * Like st-qdrant-memory's chunkExists function
     * @param {string} collectionName - Collection name
     * @param {string[]} messageIds - Array of message IDs to check
     * @param {object} filters - Multitenancy filters
     * @returns {Promise<boolean>} True if any chunk contains these message IDs
     */
    async chunkExists(collectionName, messageIds, filters = {}) {
        if (!this.baseUrl) throw new Error('Qdrant not initialized');
        if (!messageIds || messageIds.length === 0) return false;

        collectionName = this._parseCollectionName(collectionName);
        const mainCollection = collectionName;

        try {
            // Build OR conditions for message ID matching
            const should = messageIds.map(id => ({
                key: 'messageIds',
                match: { text: String(id) }
            }));

            // Also add type/sourceId filters as must conditions
            const must = [];
            if (filters.type) {
                must.push({ key: 'type', match: { value: filters.type } });
            }
            if (filters.sourceId) {
                must.push({ key: 'sourceId', match: { value: filters.sourceId } });
            }

            const filter = { should };
            if (must.length > 0) {
                filter.must = must;
            }

            const response = await this._request('POST', `/collections/${mainCollection}/points/scroll`, {
                filter,
                limit: 1,
                with_payload: false,
                with_vector: false,
            });

            return (response.result?.points?.length || 0) > 0;
        } catch (error) {
            console.error(`[Qdrant] chunkExists failed:`, error.message);
            return false;
        }
    }
}

// Export singleton instance
const qdrantBackend = new QdrantBackend();
export default qdrantBackend;
