/**
 * Step 9c: tenant scope and physical names for the new storage layouts.
 *
 *   Qdrant  mt-v2          vecthare_mt_<dim>   shared, point id = UUIDv5(tenantKey:hash)
 *   Milvus  mt-v2   (2b)   vecthare_mt_<dim>   shared, auto-id, tenantKey partition key
 *   Milvus  per-collection (2a) vh_<24 hex of sha1(tenantKey)>   one per collection
 *
 * The tenant key is computed HERE, from the SillyTavern user making the request
 * and the validated logical collection id; a client never sends it:
 *   tenantKey = <ST user handle>/<type>/<sourceId>      (max 512 characters)
 */

import path from 'node:path';
import { createHash } from 'node:crypto';
import { parseMilvusTenant } from './milvus-backend.js';

export const V2_CAPABILITY = 2;
export const MT_V2_NAME = /^vecthare_mt_([1-9]\d{0,5})$/;
export const MILVUS_OWN_NAME = /^vh_[0-9a-f]{24}$/;
export const MAX_TENANT_KEY = 512;
export const MAX_DIM = 65536;

/** Fixed VectHare namespace for point ids (never change: ids depend on it). */
export const VECTHARE_UUID_NAMESPACE = '5c3e8a5e-1f0d-4b7a-9a43-7d2e6b1c9f20';

/** RFC 4122 version-5 UUID (SHA-1), with node:crypto only. */
export function uuidv5(name, namespace = VECTHARE_UUID_NAMESPACE) {
    const ns = Buffer.from(namespace.replace(/-/g, ''), 'hex');
    if (ns.length !== 16) throw new Error('Invalid UUID namespace');
    const bytes = createHash('sha1').update(ns).update(Buffer.from(String(name), 'utf8')).digest().subarray(0, 16);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const h = bytes.toString('hex');
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

/** The SillyTavern user handle of a request, or null. */
export function userHandle(req) {
    const handle = req?.user?.profile?.handle;
    if (typeof handle === 'string' && /^[A-Za-z0-9_.-]{1,64}$/.test(handle)) return handle;
    const root = req?.user?.directories?.root;
    const folder = typeof root === 'string' ? path.basename(root) : '';
    return /^[A-Za-z0-9_.-]{1,64}$/.test(folder) ? folder : null;
}

/**
 * The scope of one logical collection for this request's user.
 * @returns {{tenantKey: string, user: string, type: string, sourceId: string, logicalId: string}|null}
 */
export function tenantFor(req, logicalId) {
    const user = userHandle(req);
    const tenant = parseMilvusTenant(logicalId);
    if (!user || !tenant) return null;
    const tenantKey = `${user}/${tenant.type}/${tenant.sourceId}`;
    if (tenantKey.length > MAX_TENANT_KEY) return null;
    return { tenantKey, user, type: tenant.type, sourceId: tenant.sourceId, logicalId };
}

export const sha256Hex = text => createHash('sha256').update(text).digest('hex');
export const configKeyForTenant = tenantKey => `vecthare_initial_${sha256Hex(tenantKey)}`;
export const milvusOwnName = tenantKey => `vh_${createHash('sha1').update(tenantKey).digest('hex').slice(0, 24)}`;
export const mtV2Name = dim => `vecthare_mt_${dim}`;

/** A saved-config slot emptied by a purge (write-once otherwise). */
export const TOMBSTONE = Object.freeze({ vecthareTombstone: 1 });
export function isTombstone(value) {
    let v = value;
    if (typeof v === 'string') { try { v = JSON.parse(v); } catch { return false; } }
    return Boolean(v) && typeof v === 'object' && v.vecthareTombstone === 1;
}

/**
 * Validates the `storage` a client names for a v2 request.
 * @returns {{layout: string, dim: number|null, physical: string|null}|{error: string}}
 */
export function parseV2Storage(backend, storage) {
    if (!storage || typeof storage !== 'object' || Array.isArray(storage)) return { error: 'storage must be an object' };
    const layouts = backend === 'qdrant' ? ['mt-v2'] : backend === 'milvus' ? ['mt-v2', 'per-collection'] : [];
    if (!layouts.includes(storage.layout)) return { error: `Unsupported ${backend} storage layout` };
    let dim = null;
    if (storage.dim !== undefined && storage.dim !== null) {
        dim = Number(storage.dim);
        if (!Number.isSafeInteger(dim) || dim < 1 || dim > MAX_DIM) return { error: 'Invalid storage dim' };
    }
    let physical = null;
    if (storage.physical !== undefined && storage.physical !== null) {
        if (typeof storage.physical !== 'string') return { error: 'Invalid storage physical name' };
        physical = storage.physical;
        const ok = storage.layout === 'mt-v2' ? MT_V2_NAME.test(physical) : MILVUS_OWN_NAME.test(physical);
        if (!ok) return { error: 'Storage physical name does not match its layout' };
        if (storage.layout === 'mt-v2') {
            const fromName = Number(MT_V2_NAME.exec(physical)[1]);
            if (dim !== null && dim !== fromName) return { error: 'Storage physical name does not match its dim' };
            dim = fromName;
        }
    }
    return { layout: storage.layout, dim, physical };
}

/**
 * Physical store of a v2 collection when it is determined without a search:
 * mt-v2 needs the vector size; Milvus per-collection depends on the tenant only.
 */
export function v2PhysicalName(layout, tenantKey, dim) {
    if (layout === 'per-collection') return milvusOwnName(tenantKey);
    if (layout === 'mt-v2' && Number.isSafeInteger(dim)) return mtV2Name(dim);
    return null;
}

/** Major.minor.patch of a version string ("v2.4.5", "2.3.0-rc1"), or null. */
export function parseVersion(text) {
    const m = /(\d+)\.(\d+)(?:\.(\d+))?/.exec(String(text || ''));
    return m ? [Number(m[1]), Number(m[2]), Number(m[3] || 0)] : null;
}
export function versionAtLeast(text, [a, b, c]) {
    const v = parseVersion(text);
    if (!v) return false;
    return v[0] !== a ? v[0] > a : v[1] !== b ? v[1] > b : v[2] >= c;
}
