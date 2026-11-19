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

// Load the Milvus SDK only if this backend is selected.
let MilvusClient;
let DataType;

/**
 * Milvus Backend Manager
 * Manages Milvus connection and operations
 */
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
        this.config = { ...this.config, ...config };
        
        console.log(`[Milvus] Initializing connection to ${this.config.address}...`);
        
        try {
            ({ MilvusClient, DataType } = await import('@zilliz/milvus2-sdk-node'));
            this.client = new MilvusClient({
                address: this.config.address,
                token: this.config.token,
                ssl: this.config.ssl
            });

            // Test connection by listing collections
            await this.client.listCollections();
            this.isConnected = true;
            console.log('[Milvus] Connection successful');
            
            // Ensure main collection exists
            // If dimensions are manually configured, use them. Otherwise default to 768.
            // If not set, it will be auto-corrected on first insert if different.
            const dim = this.config.dimensions ? Number(this.config.dimensions) : 768;
            await this.ensureCollection('vecthare_main', dim);
            
        } catch (error) {
            this.isConnected = false;
            console.error('[Milvus] Connection failed:', error.message);
            throw error;
        }
    }

    /** Read a collection property without changing the shared collection. */
    async readInitialCollectionConfig(key) {
        if (!this.isConnected || !this.client) throw new Error('Milvus not initialized');
        const info = await this.client.describeCollection({ collection_name: 'vecthare_main' });
        if (info?.status && ((info.status.code !== undefined && Number(info.status.code) !== 0) ||
            (info.status.error_code !== undefined && !['Success', 0, '0'].includes(info.status.error_code)))) {
            throw new Error(`Milvus describe collection failed: ${info.status.reason || info.status.error_code || info.status.code}`);
        }
        if (!info || typeof info !== 'object' || (!info.schema && !info.properties)) {
            throw new Error('Milvus returned an invalid collection description');
        }
        const properties = info.properties ?? info.schema?.properties ?? [];
        return Array.isArray(properties) ? properties.find(p => p.key === key)?.value
            : properties[key];
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
        if (existing !== undefined) {
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
    async ensureCollection(collectionName, vectorSize = 768) {
        if (!this.isConnected) throw new Error('Milvus not initialized');

        try {
            // Check if collection exists
            const { data: collections } = await this.client.listCollections();
            let exists = collections.find(c => c.name === collectionName);
            
            if (exists) {
                // Verify schema has 'vector' field and correct dimension
                try {
                    const desc = await this.client.describeCollection({ collection_name: collectionName });
                    const vectorField = desc.schema.fields.find(f => f.name === 'vector');
                    
                    if (!vectorField) {
                        console.warn(`[Milvus] Existing collection ${collectionName} missing 'vector' field. Dropping...`);
                        await this.client.dropCollection({ collection_name: collectionName });
                        exists = false; 
                    } else {
                        // Check dimension
                        // Milvus returns dim as string or number depending on version, safely compare
                        const existingDim = Number(vectorField.dim || vectorField.params?.find(p => p.key === 'dim')?.value);
                        
                        if (existingDim && existingDim !== vectorSize) {
                             console.warn(`[Milvus] Dimension mismatch for ${collectionName}. Existing: ${existingDim}, Required: ${vectorSize}. Dropping collection to recreate...`);
                             await this.client.dropCollection({ collection_name: collectionName });
                             exists = false;
                        }
                    }
                } catch (e) {
                    console.error('[Milvus] Describe collection failed:', e.message);
                    // If describe fails, might be safest to assume broken or let it fail later? 
                    // But we proceed for now.
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

        const mainCollection = 'vecthare_main';

        // Check dimension from first item
        const vectorSize = items[0].vector?.length || 0;
        if (vectorSize === 0) throw new Error('[Milvus] Invalid vector dimension 0');

        // Ensure collection exists
        await this.ensureCollection(mainCollection, vectorSize);

        // Prepare data matching schema
        const data = items.map(item => ({
            hash: item.hash,
            vector: item.vector,
            text: item.text,
            type: tenantMetadata.type || 'chat',
            sourceId: tenantMetadata.sourceId || 'unknown',
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
     * Delete vectors by hash
     * @param {string} collectionName 
     * @param {number[]} hashes 
     */
    async deleteVectors(collectionName, hashes) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        if (hashes.length === 0) return;

        const mainCollection = 'vecthare_main';

        try {
            if (!await this._collectionExists()) return;
            const expr = `hash in [${hashes.join(',')}]`;
            const res = await this.client.delete({
                collection_name: mainCollection,
                filter: expr
            });
            this._assertSuccess(res, 'Delete');
            console.log(`[Milvus] Deleted items with hashes: ${hashes.length}`);
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
    async countItems(filters = {}) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const expr = this._buildFilterExpr(filters) || 'hash > 0';
        if (!await this._collectionExists()) return 0;
        this._assertSuccess(await this.client.loadCollection({ collection_name: 'vecthare_main' }), 'Load');
        const res = await this.client.count({ collection_name: 'vecthare_main', expr, filter: expr });
        this._assertSuccess(res, 'Count');
        const total = Number(res?.data);
        if (!Number.isSafeInteger(total) || total < 0) throw new Error('[Milvus] Invalid count response');
        return total;
    }

    /**
     * Build boolean expression for filtering
     */
    _buildFilterExpr(filters) {
        const conditions = [];
        
        if (filters.type) conditions.push(`type == "${filters.type}"`);
        if (filters.sourceId) conditions.push(`sourceId == "${filters.sourceId}"`);
        if (filters.embeddingSource) conditions.push(`embeddingSource == "${filters.embeddingSource}"`);
        if (filters.timestampAfter) conditions.push(`timestamp >= ${filters.timestampAfter}`);
        
        return conditions.length > 0 ? conditions.join(' && ') : '';
    }

    /**
     * Query/Search for similar vectors
     */
    async queryCollection(collectionName, queryVector, topK = 10, filters = {}) {
        if (!this.isConnected) throw new Error('Milvus not initialized');

        const mainCollection = 'vecthare_main';
        const filterExpr = this._buildFilterExpr(filters);

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
    async listItems(collectionName, filters = {}, options = {}) {
        if (!this.isConnected) throw new Error('Milvus not initialized');

        const mainCollection = 'vecthare_main';
        const filterExpr = this._buildFilterExpr(filters) || 'hash > 0'; // Always need an expression for query

        try {
            if (!await this._collectionExists()) return [];
            this._assertSuccess(await this.client.loadCollection({ collection_name: mainCollection }), 'Load');

            const output_fields = ['hash', 'text', 'metadata'];
            if (options.includeVectors) output_fields.push('vector');

            // Milvus query doesn't support easy pagination like "scroll" in Qdrant
            // But we can use limit/offset
            const limit = options.limit || 100;
            const offset = options.offset || 0;

            const results = await this.client.query({
                collection_name: mainCollection,
                filter: filterExpr,
                output_fields: output_fields,
                limit: limit,
                offset: offset
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
    async getItem(collectionName, hash, filters = {}) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const mainCollection = 'vecthare_main';

        try {
            if (!await this._collectionExists()) return null;
            const results = await this.client.query({
                collection_name: mainCollection,
                filter: `hash == ${hash}`,
                output_fields: ['hash', 'text', 'vector', 'metadata'],
                limit: 1
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
    async updateItem(collectionName, hash, updates, filters = {}) {
        const existing = await this.getItem(collectionName, hash, filters);
        if (!existing) throw new Error(`Item ${hash} not found`);

        const newHash = updates.hash || hash;
        const newItem = {
            hash: newHash,
            text: updates.text || existing.text,
            vector: updates.vector || existing.vector,
            metadata: { ...existing.metadata, ...updates }
        };

        if (String(newHash) !== String(hash)) {
            // Write the replacement first; remove the old hash only afterwards.
            await this.insertVectors(collectionName, [newItem], filters);
            await this.deleteVectors(collectionName, [hash]);
            return;
        }
        // Milvus insert does not replace an existing key: remove, add, and
        // restore the original if the add fails.
        await this.deleteVectors(collectionName, [hash]);
        try {
            await this.insertVectors(collectionName, [newItem], filters);
        } catch (error) {
            await this.insertVectors(collectionName, [existing], filters).catch(() => {});
            throw error;
        }
    }

    /**
     * Purge collection (drop & recreate or delete by filter)
     * If filters present, delete by filter. If no filters, drop collection.
     */
    async purgeCollection(collectionName, filters = {}) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const mainCollection = 'vecthare_main';

        const hasFilters = Object.keys(filters).length > 0;

        try {
            if (hasFilters) {
                const expr = this._buildFilterExpr(filters);
                if (expr) {
                    const res = await this.client.delete({
                        collection_name: mainCollection,
                        filter: expr
                    });
                    this._assertSuccess(res, 'Purge');
                    console.log(`[Milvus] Purged items with filter: ${expr}`);
                }
            } else {
                const exists = await this.client.hasCollection({ collection_name: mainCollection });
                if (exists?.value === false) return; // nothing to drop
                const res = await this.client.dropCollection({ collection_name: mainCollection });
                this._assertSuccess(res, 'Drop collection');
                console.log(`[Milvus] Dropped collection ${mainCollection}`);
                // Re-create is handled by next insert/init
            }
        } catch (error) {
            console.error('[Milvus] Purge failed:', error.message);
            throw error; // a failed purge must not be reported as a confirmed deletion
        }
    }

    /**
     * Get stats
     */
    async getCollectionStats(collectionName, filters = {}) {
        if (!this.isConnected) throw new Error('Milvus not initialized');
        const mainCollection = 'vecthare_main';

        try {
            const stats = await this.client.getCollectionStatistics({
                collection_name: mainCollection
            });

            const rowCount = stats.data.row_count;

            // Approximate token count if we can't query all efficiently
            // For now return basic stats
            return {
                chunkCount: Number(rowCount),
                backend: 'milvus'
            };

        } catch (error) {
            return { chunkCount: 0, backend: 'milvus', error: error.message };
        }
    }
}

const milvusBackend = new MilvusBackend();
export default milvusBackend;
