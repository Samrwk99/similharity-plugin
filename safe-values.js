/**
 * Validation helpers shared by the Similharity backends.
 *
 * Request bodies are interpolated into Milvus / LanceDB filter strings, so every
 * value that reaches a filter is validated here first. Anything that is not
 * provably safe is REJECTED (never "cleaned up"): a request that cannot be
 * validated must not run in a widened or altered form.
 */

/** Largest number of hashes accepted in one filter expression. */
export const MAX_HASHES_PER_REQUEST = 50000;

/**
 * Convert one hash to a canonical integer string, or null when it is not a
 * safe integer. Accepts numbers and plain base-10 integer strings only
 * ("12", "-12"); rejects "1 OR 1=1", "1e3", "0x10", "", NaN, floats, etc.
 * @param {unknown} value
 * @returns {string|null}
 */
export function toSafeHash(value) {
    if (typeof value === 'number') {
        return Number.isSafeInteger(value) ? String(value) : null;
    }
    if (typeof value === 'string' && /^-?\d{1,16}$/.test(value)) {
        const n = Number(value);
        return Number.isSafeInteger(n) ? String(n) : null;
    }
    return null;
}

/**
 * Validate a whole hash list. Throws (with `status = 400`) when the input is
 * not an array, is too large, or contains ANY unsafe entry, so a crafted body
 * can never widen a delete by smuggling one bad element past the rest.
 * @param {unknown} hashes
 * @returns {string[]} canonical integer strings
 */
export function toSafeHashList(hashes) {
    if (!Array.isArray(hashes)) throw badRequest('hashes must be an array');
    if (hashes.length > MAX_HASHES_PER_REQUEST) throw badRequest('too many hashes in one request');
    const out = new Array(hashes.length);
    for (let i = 0; i < hashes.length; i++) {
        const safe = toSafeHash(hashes[i]);
        if (safe === null) throw badRequest('hashes must be safe integers; request rejected');
        out[i] = safe;
    }
    return out;
}

/** Same as toSafeHash but throws a 400 error. */
export function requireSafeHash(value) {
    const safe = toSafeHash(value);
    if (safe === null) throw badRequest('hash must be a safe integer');
    return safe;
}

/**
 * A string that may be placed inside a double-quoted filter literal after
 * `quoteLiteral` escaping: non-empty, bounded, and free of control characters.
 * Used for collection identities, which for older collections can legitimately
 * contain spaces, `@`, accents etc. (e.g. a chat file name).
 * @param {unknown} value
 * @param {number} maxLength
 * @returns {boolean}
 */
export function isValidLiteral(value, maxLength = 128) {
    return typeof value === 'string' && value.length >= 1 && value.length <= maxLength &&
        !/[\u0000-\u001f\u007f\u2028\u2029]/.test(value);
}

/**
 * Double-quoted, escaped filter literal. The only way a request-supplied
 * string may reach a filter expression.
 * @param {string} value validated with isValidLiteral
 * @returns {string}
 */
export function quoteLiteral(value) {
    return '"' + value.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
}

/**
 * Strict whitelist for values that are identifiers by nature (provider ids):
 * letters, digits, `_ . : -`, 1..maxLength characters.
 * @param {unknown} value
 * @param {number} maxLength
 * @returns {boolean}
 */
export function isSafeFilterString(value, maxLength = 128) {
    return typeof value === 'string' && value.length >= 1 && value.length <= maxLength &&
        /^[A-Za-z0-9_.:-]+$/.test(value);
}

export function badRequest(message) {
    const error = new Error(message);
    error.status = 400;
    return error;
}
