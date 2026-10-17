/** Dependency-free Qdrant contract tests. Run: node tests/p0-tenant-isolation.mjs */
import assert from 'node:assert/strict';
import qdrant from '../qdrant-backend.js';

qdrant.baseUrl = 'http://mock-qdrant';
const collection = 'vecthare_multitenancy';
const points = new Map();
let deletes = [];
let requestLog = [];
let collectionDimension = 3;
function matches(point, filter = {}) {
  return [...(filter.must || [])].every(({key,match}) => match?.any ? match.any.includes(point.payload[key]) : point.payload[key] === match?.value) &&
    [...(filter.must_not || [])].every(({key,match}) => point.payload[key] !== match?.value);
}
qdrant._request = async (verb, path, body = {}) => {
  requestLog.push({verb,path,body});
  if (verb === 'GET' && path === '/collections') return {result:{collections:[{name:collection}]}};
  if (verb === 'GET' && path === `/collections/${collection}`) return {result:{config:{params:{vectors:{size:collectionDimension}}},points_count:points.size}};
  if (verb === 'PUT' && path === `/collections/${collection}/points?wait=true`) {
    for (const point of body.points) points.set(point.id, structuredClone(point));
    return {result:{status:'completed'}};
  }
  if (verb === 'POST' && path === `/collections/${collection}/points/search`) {
    return {result: [...points.values()].filter(p=>matches(p,body.filter)).map(p=>({payload:p.payload,score:.9,id:p.id})).slice(0,body.limit)};
  }
  if (verb === 'POST' && path === `/collections/${collection}/points/scroll`) {
    return {result:{points: [...points.values()].filter(p=>matches(p,body.filter)).map(p=>({...p})),next_page_offset:null}};
  }
  if (verb === 'POST' && path === `/collections/${collection}/points/delete?wait=true`) {
    deletes.push(body);
    for (const [id,point] of points) if (body.filter ? matches(point,body.filter) : body.points.includes(id)) points.delete(id);
    return {result:{status:'completed'}};
  }
  if (verb === 'POST' && path === `/collections/${collection}/points/payload?wait=true`) {
    for (const [id,point] of points) if (body.filter ? matches(point,body.filter) : body.points.includes(id)) points.set(id,{...point,payload:{...point.payload,...body.payload}});
    return {result:{status:'completed'}};
  }
  if (verb === 'DELETE') throw new Error('Unsafe physical collection deletion attempted');
  return {result:{status:'completed'}};
};
let count = 0;
function ok(name){ count++; console.log('PASS',name); }
const item=(hash,text)=>({hash,text,vector:[0.1,0.2,0.3], metadata:{importance:8}});
const filtA={content_type:'chat-A'};
const filtB={content_type:'chat-B'};
await qdrant.insertVectors(collection,[item(123,'same hash from A')],filtA);
await qdrant.insertVectors(collection,[item(123,'same hash from B')],filtB);
assert.equal(points.size,2);
const ids=[...points.keys()];
assert.equal(ids.every(x=>typeof x==='string' && /^[0-9a-f-]{36}$/.test(x)),true);
assert.notEqual(ids[0],ids[1]);
assert.equal([...points.values()].find(p=>p.payload.content_type==='chat-A').payload.hash,123);
ok('same hash two tenants receives different stable UUID point IDs');
await qdrant.insertVectors(collection,[item(123,'updated A')],filtA);
assert.equal(points.size,2);
assert.equal([...points.values()].find(p=>p.payload.content_type==='chat-B').payload.text,'same hash from B');
ok('upsert of A cannot overwrite B');
assert.equal((await qdrant.listItems(collection,filtA)).length,1);
assert.equal((await qdrant.listItems(collection,filtB)).length,1);
assert.equal((await qdrant.queryCollection(collection,[0.1,0.2,0.3],10,filtA)).length,1);
assert.equal((await qdrant.queryCollection(collection,[0.1,0.2,0.3],10,filtB))[0].text,'same hash from B');
ok('filtered list and search operate server-side with tenant namespace');
// Legacy numeric points remain readable and filtered, including ones with colliding hashes.
points.set(999,{id:999,vector:[.1,.2,.3],payload:{hash:999,content_type:'chat-B',text:'legacy B',type:'chat'}});
assert.equal((await qdrant.listItems(collection,filtA)).length,1);
assert.equal((await qdrant.listItems(collection,filtB)).length,2);
ok('old shared-collection numeric points remain tenant-filtered');
await qdrant.deleteVectors(collection,[123,999],filtA);
assert.equal([...points.values()].filter(p=>p.payload.content_type==='chat-A').length,0);
assert.equal([...points.values()].filter(p=>p.payload.content_type==='chat-B').length,2);
ok('delete-by-hash cannot delete equal hash belonging to B');
await qdrant.insertVectors(collection,[item(123,'A reinserted')],filtA);
const a=await qdrant.getItem(collection,123,filtA);
assert.equal(a.text,'A reinserted');
await qdrant.updateItemMetadata(collection,123,{importance:40},filtA);
assert.equal([...points.values()].find(p=>p.payload.content_type==='chat-A').payload.importance,40);
assert.equal([...points.values()].find(p=>p.payload.content_type==='chat-B').payload.importance,8);
await assert.rejects(qdrant.updateItemMetadata(collection,123,{content_type:'chat-B'},filtA),/protected/);
ok('tenant-scoped item lookup and metadata edit do not change ownership');
await qdrant.updateItem(collection,123,{hash:456,text:'A replaced',vector:[.1,.2,.3]},filtA);
assert.equal((await qdrant.listItems(collection,filtA))[0].hash,456);
assert.equal((await qdrant.listItems(collection,filtB)).length,2);
ok('text replacement inserts first and deletes only old tenant hash');
await qdrant.purgeCollection(collection,filtA);
assert.equal((await qdrant.listItems(collection,filtA)).length,0);
assert.equal((await qdrant.listItems(collection,filtB)).length,2);
assert.equal(deletes.at(-1).filter.must.some(c=>c.key==='content_type' && c.match.value==='chat-A'),true);
ok('purge tenant A preserves tenant B and physical collection');
const remaining=[...points.keys()];
collectionDimension=5;
await assert.rejects(qdrant.insertVectors(collection,[item(777,'incompatible')],filtA),/dimension|Dimension/i);
assert.deepEqual([...points.keys()],remaining);
assert.equal(requestLog.some(x=>x.verb==='DELETE'),false);
ok('dimension mismatch never drops shared collection');
console.log(`\n${count} Similharity tenant isolation tests passed`);
